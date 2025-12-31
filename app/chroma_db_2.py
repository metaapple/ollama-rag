# chroma_db.py
from __future__ import annotations

import re
import uuid
from typing import List, Optional, Dict, Any, Tuple

import requests
import chromadb
import fitz  # PyMuPDF


class ChromaRAG:
    """
    ChromaDB(영속) + Ollama(임베딩/생성) 기반 최소 RAG 엔진 (개선판)

    개선 포인트:
    - Google 접두사로 RAG on/off 하지 않고, 기본은 always-retrieve 후 근거 부족 시 fallback
    - source 필터(where) 지원: 문서가 섞여도 특정 문서만 검색 가능
    - 문장/단락 기반 chunking
    - collection.add 배치 처리
    - 컨텍스트 구성 개선
    """

    def __init__(
        self,
        chroma_dir: str = "./chroma_data",
        collection_name: str = "rag_docs",
        ollama_base_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        gen_model: str = "gemma3:1b",
        timeout: int = 120,
    ):
        # Ollama 설정
        self.ollama_base_url = ollama_base_url.rstrip("/")
        self.embed_model = embed_model
        self.gen_model = gen_model
        self.timeout = timeout

        # Chroma 설정
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def __str__(self):
        return f"{self.client} embed={self.embed_model} gen={self.gen_model} collection={self.collection}"

    # -----------------------------
    # 1) Ollama: Embedding / Generate
    # -----------------------------
    def embed(self, text: str) -> List[float]:
        """
        단일 텍스트 임베딩
        """
        text = (text or "").strip()
        if not text:
            return []

        url = f"{self.ollama_base_url}/api/embeddings"
        resp = requests.post(
            url,
            json={"model": self.embed_model, "prompt": text},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("embedding", [])

    def generate(self, prompt: str) -> str:
        """
        단일 프롬프트 생성
        """
        prompt = (prompt or "").strip()
        if not prompt:
            return ""

        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.gen_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # RAG 답변 안정성 우선이면 낮추는 편이 유리
                "num_predict": 256,  # 너무 짧으면 근거를 다 쓰기 전에 끊길 수 있어 상향
            },
        }
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()

    # -----------------------------
    # 2) Text -> Chunks
    # -----------------------------
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """
        한국어/영어 혼합 문서에서 대략적인 문장 단위 분리.
        - 줄바꿈, 마침표, 물음표, 느낌표 기준으로 문장 경계를 잡음
        """
        text = (text or "").strip()
        if not text:
            return []

        # 줄바꿈은 문장 경계로 취급
        text = re.sub(r"\n{2,}", "\n", text)

        # 문장 분리: 종결 부호 뒤 공백/줄바꿈을 기준으로 split
        parts = re.split(r"(?<=[\.\?\!。！？])\s+|\n+", text)
        parts = [p.strip() for p in parts if p and p.strip()]
        return parts

    @classmethod
    def chunk_text(
        cls,
        text: str,
        max_chars: int = 1200,
        overlap_chars: int = 150,
    ) -> List[str]:
        """
        문장 기반으로 묶어서 chunk 생성.
        - max_chars를 넘기기 직전까지 문장을 누적
        - overlap_chars는 "문장 단위로" 마지막 일부를 다음 chunk에 일부 포함시키는 방식으로 근사
        """
        sents = cls._split_sentences(text)
        if not sents:
            return []

        chunks: List[str] = []
        buf: List[str] = []
        buf_len = 0

        def flush_buffer():
            nonlocal buf, buf_len
            if not buf:
                return
            chunk = " ".join(buf).strip()
            if chunk:
                chunks.append(chunk)

            # overlap: 마지막 overlap_chars 만큼의 tail을 다음 버퍼로 가져오기
            if overlap_chars > 0 and chunk:
                tail = chunk[-overlap_chars:].strip()
                buf = [tail] if tail else []
                buf_len = len(tail)
            else:
                buf = []
                buf_len = 0

        for s in sents:
            if not s:
                continue

            # 문장 하나가 max_chars보다 큰 경우: 강제 분할 (최후 수단)
            if len(s) > max_chars:
                # 버퍼 먼저 flush
                flush_buffer()
                start = 0
                while start < len(s):
                    end = min(start + max_chars, len(s))
                    piece = s[start:end].strip()
                    if piece:
                        chunks.append(piece)
                    if end == len(s):
                        break
                    start = max(0, end - overlap_chars)
                continue

            # 버퍼에 넣을 수 있으면 누적
            if buf_len + len(s) + 1 <= max_chars:
                buf.append(s)
                buf_len += len(s) + 1
            else:
                flush_buffer()
                buf.append(s)
                buf_len = len(s) + 1

        flush_buffer()
        return chunks

    # -----------------------------
    # 3) PDF -> Text
    # -----------------------------
    @staticmethod
    def pdf_to_text(pdf_bytes: bytes) -> str:
        """
        텍스트 기반 PDF: 잘 동작
        이미지 스캔본: 텍스트가 거의 안 나올 수 있음(별도 OCR 필요)
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts = []
        for page in doc:
            parts.append(page.get_text("text"))
        return "\n".join(parts).strip()

    # -----------------------------
    # 4) Ingest
    # -----------------------------
    def count(self) -> int:
        return self.collection.count()

    def get_collection(self, limit: int = 30):
        return self.collection.get(include=["documents", "metadatas"], limit=limit)

    def ingest_document(
        self,
        raw_text: str,
        source: str,
        max_chars: int = 1200,
        overlap_chars: int = 150,
        meta_extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        """
        텍스트 -> chunk -> embedding -> chroma 저장
        개선: add를 chunk마다 호출하지 않고 배치로 1회 호출
        """
        raw_text = (raw_text or "").strip()
        if not raw_text:
            return 0

        meta_extra = meta_extra or {}
        chunks = self.chunk_text(raw_text, max_chars=max_chars, overlap_chars=overlap_chars)
        if not chunks:
            return 0

        ids: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []

        for i, ch in enumerate(chunks):
            emb = self.embed(ch)
            if not emb:
                continue
            ids.append(str(uuid.uuid4()))
            embeddings.append(emb)
            metadatas.append({"chunk": i, "source": source, **meta_extra})

        if not ids:
            return 0

        self.collection.add(
            ids=ids,
            documents=chunks[: len(ids)],
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(ids)

    # -----------------------------
    # 5) Retrieval
    # -----------------------------
    def query_docs(
        self,
        question: str,
        top_k: int = 10,
        source: Optional[str] = None,
        include_meta: bool = False,
    ) -> Tuple[List[str], Optional[List[Dict[str, Any]]]]:
        """
        question 임베딩 -> chroma query
        - source를 주면 해당 source(문서)만 검색 (중요)
        """
        n = self.count()
        if n <= 0:
            return ([], [] if include_meta else None)

        top_k = max(1, min(top_k, n))
        q_emb = self.embed(question)
        if not q_emb:
            return ([], [] if include_meta else None)

        where = {"source": source} if source else None

        # Chroma의 query 반환 구조는 include 설정/버전에 따라 차이가 있을 수 있어 방어적으로 처리
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas"] if include_meta else ["documents"],
        )

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0] if include_meta else None
        return docs, metas

    # -----------------------------
    # 6) RAG Answering (always-retrieve + fallback)
    # -----------------------------
    @staticmethod
    def _strip_google_prefix(question: str) -> str:
        """
        'Google ...' 접두사를 안전하게 제거 (데모 호환용)
        """
        q = (question or "").strip()
        # 정확히 "Google" 단어로 시작할 때만 제거
        q = re.sub(r"^\s*Google\s+", "", q, flags=re.IGNORECASE).strip()
        return q

    def ask(
        self,
        question: str,
        top_k: int = 10,
        source: Optional[str] = None,
        force_rag: bool = False,
        min_context_chars: int = 200,
        max_context_chars: int = 3500,
        context_top_n: int = 6,
    ) -> Dict[str, Any]:
        """
        기본 정책:
        - 항상 retrieval 수행
        - 검색 결과가 충분하면 RAG
        - 검색 결과가 빈약하면 일반 생성으로 fallback
        - force_rag=True면 문서가 빈약해도 RAG 강제
        """
        original_q = (question or "").strip()
        if not original_q:
            return {"answer": "", "chroma-db": [], "mode": "입력 없음"}

        q = self._strip_google_prefix(original_q)

        # 1) retrieval
        docs, metas = self.query_docs(q, top_k=top_k, source=source, include_meta=True)

        # 2) 컨텍스트 구성: 상위 context_top_n개만 사용 + 전체 길이 제한
        selected_docs: List[str] = []
        total = 0
        for d in docs[: max(1, context_top_n)]:
            d = (d or "").strip()
            if not d:
                continue
            # 청크 단위로 길이 제한 적용
            if total + len(d) > max_context_chars:
                break
            selected_docs.append(d)
            total += len(d)

        context = "\n\n---\n\n".join(selected_docs).strip()

        # 3) RAG vs fallback 판단
        enough_context = len(context) >= min_context_chars

        if not enough_context and not force_rag:
            # fallback: 일반 생성
            answer = self.generate(q)
            return {
                "answer": answer,
                "chroma-db": [],
                "mode": "일반 생성 (fallback: 문서 근거 부족)",
                "source_filter": source,
            }

        if not context:
            # RAG 강제인데도 문서가 없으면
            return {
                "answer": "문서에서 찾을 수 없습니다.",
                "chroma-db": [],
                "mode": "RAG (문서 없음)",
                "source_filter": source,
            }

        prompt = f"""너는 문서 기반 QA 어시스턴트다.
아래 CONTEXT에 있는 정보만 사용해서 질문에 답하라.
정보가 없으면 '문서에서 찾을 수 없습니다.'라고 답하라.

[CONTEXT]
{context}

[QUESTION]
{q}

[ANSWER]
"""
        answer = self.generate(prompt)
        return {
            "answer": answer,
            "chroma-db": selected_docs,
            "mode": "RAG (Chroma DB 검색)",
            "source_filter": source,
            "used_context_chars": len(context),
        }


if __name__ == "__main__":
    rag = ChromaRAG()

    # 예시 1) 문서 적재
    text = "우리의 리눅스 서버 프로젝트명은 9999이다."
    rag.ingest_document(text, source="test3.txt")

    # 예시 2) 특정 source만 검색 (섞임 방지)
    print(rag.ask("우리의 리눅스 서버 프로젝트명은?", top_k=10, source="test3.txt"))

    # 예시 3) 데모 호환: Google 접두사도 동작(그냥 제거해서 처리)
    print(rag.ask("Google 우리의 리눅스 서버 프로젝트명은?", top_k=10, source="test3.txt"))
