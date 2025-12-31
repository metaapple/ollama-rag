# chroma_db.py
from __future__ import annotations

import uuid
from typing import List, Optional, Dict, Any

import requests
import chromadb
from prompt_toolkit.renderer import print_formatted_text

import fitz  # PyMuPDF
from sympy.multipledispatch.dispatcher import source


class ChromaRAG:
    ###############
    # 1. 설정부분.
    # ollama설정
    # 올라마 url : http://localhost:11434
    # 인베딩 모델 : nomic-embed-text
    # 생성형 모델 : llama3:1b, gemma3:1bf

    def __init__(self,
                 chroma_dir: str = './chroma_data',
                 collection_name: str = 'rag_docs',
                 ollama_base_url: str = 'http://localhost:11434',
                 embed_model: str = 'nomic-embed-text',
                 gen_model: str = 'gemma3:1b'):

        # Ollama 설정
        self.ollama_base_url = ollama_base_url
        self.embed_model = embed_model
        self.gen_model = gen_model

        # chroma설정
        # 폴더만든 것에 chroma db연결함.
        # --> chroma_data
        # collection(table, 폴더)를 생성함.
        # --> rag_docs
        self.client = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)  # rag_docs
        self.collection2 = self.client.get_or_create_collection(name=collection_name + str(2))  # rag_docs2

    def __str__(self):
        return str(self.client) + " " + self.embed_model + " " + self.gen_model + " " + str(
            self.collection) + " " + str(self.collection2)

    ###############
    # 2. 임베딩하고 ollama요청해서 답변받아오는 부분

    # 임베딩 embed

    def embed(self, text: str) -> List[float]:  # 리턴타입!!
        # ollama에 주소로 임베딩해달라고 요청합시다.
        url = self.ollama_base_url + '/api/embeddings'
        resp = requests.post(url, json={'model': self.embed_model, 'prompt': text}, timeout=120)
        # print(resp.json())  # dict형태로 만들어서 프린트.
        data = resp.json()  # {"embedding" : [0.1232, 0.234324]}
        return data['embedding']

    # 답생성 generate
    def generate(self, prompt: str) -> str:
        url = self.ollama_base_url + '/api/generate'
        payload = {
            "model": self.gen_model,  # gemma3:1b
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,  # 네가 찾은 것 중에서 아주 정확한 것만!!
                "num_predict": 64,
                # 짧게, 한글은 한글자에 2바이트임, 30글자 정도, 문장 2-3개
                # 환경에 따라 지원되는 옵션이 다를 수 있음
                # "repeat_penalty": 1.1,
            }
        }
        r = requests.post(url, json=payload, timeout=120)
        print(r.json())
        data = r.json()
        return data['response']

    ###############
    # 3. chuck만드는 부분

    # 통으로 읽은 text를 작게 자르자(chuck, 조각)
    @staticmethod
    def chunk_text(text: str, max_chars: int = 500, overlap_chars: int = 100) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        chunks = []
        start = 0
        n = len(text)

        while start < n:
            end = min(start + max_chars, n)
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            if end == n:
                break

            start = max(0, end - overlap_chars)

        return chunks

    # pdf를 읽어서 text로 만들자.
    @staticmethod
    def pdf_to_text(pdf_bytes: bytes) -> str:
        """
        PDF 바이너리(bytes)를 받아서 전체 텍스트를 추출합니다.
        - 스캔본(이미지) PDF는 텍스트가 거의 안 나올 수 있음(OCR 필요)
        """
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        parts = []
        for page in doc:
            parts.append(page.get_text("text"))
        return "\n".join(parts).strip()

    # collection에 몇 개 들어있는지 확인하는 함수
    def count(self) -> int:
        return self.collection.count()

    def get_collection(self):
        results = self.collection.get(include=["documents", "metadatas"], limit=30)
        return results

    ###############
    # 4. 크로마 db에 적재하는 부분
    # 텍스트를 크로마db에 적재하자.(ingest)
    def ingest_texts(self, texts: List[str], source: str = 'manual') -> int:
        if not texts:
            return 0
        for i, t in enumerate(texts):
            self.collection.add(
                ids=[str(uuid.uuid4())],
                documents=[t],
                embeddings=[self.embed(t)],
                metadatas=[{"chunk": i, "source": source}],
            )
        return len(texts)

    ###############
    # 5. 텍스트를 읽어서 청크--> 임베딩 --> 크로마db에 저장
    def ingest_document(
            self,
            raw_text: str,
            source: str,
            max_chars: int = 1200,
            overlap_chars: int = 150,
            meta_extra: Optional[Dict[str, Any]] = None,
    ) -> int:
        meta_extra = meta_extra or {}
        chunks = self.chunk_text(raw_text, max_chars=max_chars, overlap_chars=overlap_chars)
        if not chunks:
            return 0

        for i, ch in enumerate(chunks):
            self.collection.add(
                ids=[str(uuid.uuid4())],
                documents=[ch],
                embeddings=[self.embed(ch)],
                metadatas=[{"chunk": i, "source": source, **meta_extra}],
            )
        return len(chunks)


    ### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    ### @@@@@@@@@@@@@@@@@@@ 크로마db에서 검색 @@@@@@@@@@@@@@@@@@@@@@
    ### 질문 --> 임베딩 --> 크로마db에서 검색
    ### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    def query_docs(self, question: str, top_k: int = 10) -> List[str]:
        n = self.count()
        if n <= 0:
            return []

        top_k = min(top_k, n)
        q_emb = self.embed(question)

        res = self.collection.query(query_embeddings=[q_emb], n_results=top_k)
        docs = (res.get("documents") or [[]])[0]
        return docs

    ### @@@@@@@@@@@@@@@@@@@ 크로마db + gemma3:1b에서 검색 @@@@@@@@@@@@@@@@@@@@@@
    ### Google붙으면 질문 --> 임베딩 --> 크로마db에서 검색
    ### 안붙으면 질문 --> 프롬프트 --> 올라마 gemma3:1b에서 생성
    ### @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def ask(self, question: str, top_k: int = 10) -> Dict[str, Any]:
        import re

        # 1. 질문이 "Google" 또는 "google" 등으로 시작하는지 확인 (앞뒤 공백 무시)
        if re.match(r"^\s*Google", question, flags=re.IGNORECASE):
            # ---- RAG 모드 (Chroma DB 검색 후 generate) ----
            docs = self.query_docs(question.lstrip("Google ").strip(), top_k=top_k)  # "Google" 부분 제거 후 검색
            if not docs:
                answer = "문서가 존재하지 않습니다."
                return {"answer": answer, "chroma-db": [], "mode": "RAG (문서 없음)"}

            context = "\n\n---\n\n".join(docs)[:3000]
            prompt = f"""너는 문서 기반 QA 어시스턴트다.
    아래 CONTEXT에 있는 정보만 사용해서 질문에 답하라.
    정보가 없으면 '문서에서 찾을 수 없습니다.'라고 답하라.

    [CONTEXT]
    {context}

    [QUESTION]
    {question.lstrip("Google ").strip()}

    [ANSWER]
    """
            answer = self.generate(prompt).strip()
            return {"answer": answer, "chroma-db": docs, "mode": "RAG (Chroma DB 검색)"}

        else:
            # ---- 일반 생성 모드 (Chroma DB 검색 없이 바로 gemma3 호출) ----
            prompt = question  # 그대로 전달하거나 필요 시 시스템 프롬프트 추가 가능
            answer = self.generate(prompt).strip()
            return {"answer": answer, "chroma-db": None, "mode": "일반 생성 (gemma3)"}


if __name__ == '__main__':
    rag = ChromaRAG()  # def __init__() 호출됨.
    #     # print(rag) # def __str__()호출됨.
    #     # rag.embed(text='hello')
    #     # rag.embed(text='world')
    #     # print(rag.generate(prompt="저녁 뭐먹냐.."))
    #     # print(rag.generate(prompt="한식"))
    #     text = "세련된 이미지와 탄탄한 연기력으로 사랑받아온 배우 이하늬 씨가 의외의 소식으로 대중을 놀라게 했습니다.본인이 사내이사로 재직 중인 법인을 '대중문화예술기획업'으로 정식 등록하지 않고 운영했다는 혐의를 받고 있는데요.서울 강남경찰서는 지난 23일, 이하늬 씨와 현재 해당 법인의 대표를 맡고 있는 남편 장 모 씨를 검찰에 불구속 송치했습니다.문제가 된 '대중문화산업법'[출처] 이하늬, '미등록 기획사' 운영 혐의로 검찰 송치... 소속사 측 해명은?|작성자 Catsle"
    #     result = ChromaRAG.chunk_text(text) #list[str] ==> ['skdk', 'dsdf', ...]
    #     print(result)
    #
    # # # embedding test
    # #     result2 = rag.embed(text = result[0])
    # #     print(result2)
    # #
    # # # gemma3:1b test
    # #     print(rag.generate(prompt=text))
    #
    # # chunk한거 chromadb에 적재
    #     result3 = rag.ingest_texts(result)
    #     print(result3)
    #
    # 적재한거 몇개인지 test
    # print(rag.count())
    # print(rag.get_collection())

# text1 = "두 사람은 피부 관리에 앞서 책과 일상적인 화제에 대해 대화를 나누던 중이었고, 이 과정에서 기안84는 고주파 기계를 바라보며 '앞에서는 되게 아름다운 이야기를 하다가 갑자기 이걸 꺼내니까 가정 방문 야매 치료사 같다고 농담을 던졌다.당시 발언은 가벼운 농담으로 받아들여지며 별다른 논란 없이 지나갔지만 최근 박나래가 지인으로 알려진 이른바 주사 이모'에게 받은 불법 의료 시술 사건이 공론화되면서 해당 방송 장면이 다시 소환됐다.누리꾼들은 지금 보니 진짜 의미심장하다, 우연이라고 볼 수 없는 묘한 발언, 괜히 다시 떠오르고 있는 장면이 아니다. 이미 뭔고 알고 있었던 것 이라는 반응을 보인다.한편 박나래는 최근 '주사 이모' 의혹과 더불어 전 매니저들과의 갈등이 수면 위로 드러나며 여러 논란에 휩싸였다. 박나래는 불법 의료 시술 의혹과 전 매니저에 대한 갑질 의혹 등이 잇따라 제기되자 지난 8일 방송 활동을 모두 중단했다. 현재는 법적 절차에 따라 대응하겠다는 방침을 유지하고 있다."
# text2 = "웹툰 작가 겸 방송인 기안84의 과거 발언이 최근 코미디언 박나래를 둘러싼 이른바 주사 이모 논란과 맞물리며 주목받고 있다.25일 연예계에 따르면 최근 여러 온라인 커뮤니티와 SNS에서는 약 4개월 전 공개된 기안84의 유튜브 채널 '인생84' 영상 일부가 다시 확산되고 있다.문제의 장면은 배우 이세희가 출연한 편으로, 당시 기안84가 이세희의 자택을 방문해 대화를 나누는 과정에서 나온 한 발언이다.영상에서 이세희는 집에 놀러 오면 직접 해준다며 평소 즐겨 한다는 피부 관리를 소개했고, 중고 플랫폼을 통해 약 200만 원에 구매했다는 고주파 피부 관리 기계를 꺼내 기안84에게 직접 시연에 나섰다."
# text3 = "이날 첫인상 투표 결과 용담과 국화는 1표, 백합은 3표였다. 표를 받지 못한 장미와 튤립. 장미는 '창피하고, 서운하기도 했다. 분발해야겠단 생각을 했다' 했고, 튤립은 '친구들이 엄청 놀리겠다 싶었다. 엄마한테 뭐라 하지. (엄마가) 놀리기도 하고, 진심으로 실망할 것 같다고 걱정했다. 5분 대화 후 백합은 22기 상철로 마음이 바뀌었다.28기 영수는 자기 소개한 지 얼마 안 됐다면서 84년생이며, 스타트업 창업가라고 밝혔다. 영수는 돌싱인데, 2년 정도 됐다. 아이는 없다라며 지금까지 거짓말치는 인생은 살지 않았다라며 '나는 솔로' 방영 후 시청자들의 반응을 의식했다. 영수는 방송 보면서 부족한 점을 많이 느꼈다. 선입견이 있다면 선입견을 내려놓고 대화를 나누었으면 좋겠다고 전했다."
#
# rag = ChromaRAG()
# list = [text1, text2, text3]
#
# for text in list:
#     result = ChromaRAG.chunk_text(text)  # list[str] ==> ['skdk', 'dsdf', ...]
#     print(result)
#     result3 = rag.ingest_texts(result)
#     print(result3)
#     print(rag.count())
#     print(rag.get_collection())
#
#     print("======================================")


###### 실습 1
    num = rag.ingest_document("2026년 우리회사의 목표는 킹왕짱이 된다.", "test2")
    print(num)

    chroma_result = rag.ask("Google 2026년 우리회사의 목표는", 10)

    print(chroma_result)

##### 실습 2
    # num = rag.ingest_document("우리의 리눅스 서버 비밀번호는 9999이다.", "test")
    # print(num)

    chroma_result = rag.ask("Google 우리의 리눅스 서버 비밀번호는 ", 4)

    print(chroma_result)

##### 실습 3
    # num = rag.ingest_document("거주지는 영등포", "test")
    # print(num)
    #
    chroma_result = rag.ask("Google 거주지는 ", 4)

    print(chroma_result)

##### 실습 4
    # swagger에서 pdf파일 첨부 후
    chroma_result = rag.ask("Google 가장 큰 병목은 ", 20)

    print(chroma_result)