# schemas.py
from __future__ import annotations
# ↑ (선택) 타입 힌트 평가를 뒤로 미루는 옵션입니다.
#   파이썬 버전에 따라 타입 관련 오류를 줄이는 데 도움이 됩니다.
#   지금 코드에서는 "호환성/안정성" 목적에 가깝습니다.

from typing import List, Optional
# ↑ List: 여러 개의 값을 담는 리스트 타입 힌트 (예: List[str])
#   Optional: 값이 있을 수도/없을 수도 있음 (예: Optional[str]는 str 또는 None)

from pydantic import BaseModel
# ↑ FastAPI는 요청/응답 JSON을 자동으로 검증해야 합니다.
#   Pydantic의 BaseModel을 상속하면,
#   - 요청 데이터가 어떤 모양이어야 하는지(스키마)
#   - 타입이 맞는지 검사
#   - 기본값이 있으면 자동으로 채움
#   을 해줍니다.


# =========================================================
# 1) /ingest_texts 요청 바디(JSON) 형태를 정의
# =========================================================
class IngestTextsRequest(BaseModel):
    # texts: 사용자가 넣고 싶은 문서 텍스트 목록
    # 예) {"texts": ["문서1", "문서2", "문서3"], "source": "manual"}
    texts: List[str]

    # source: 문서의 출처(라벨) 같은 것
    # 기본값이 "manual"이라서 사용자가 안 보내면 자동으로 manual로 들어갑니다.
    # 예) source="manual", source="huggingface", source="pdf" 같은 식으로 구분 가능
    source: str = "manual"


# =========================================================
# 2) /ingest_hf 요청 바디(JSON) 형태를 정의
#    Hugging Face에서 파일을 다운로드해서 문서로 넣을 때 사용
# =========================================================
class IngestHFRequest(BaseModel):
    # repo_id: 허깅페이스 리포지토리 이름
    # 예) "gpt2"
    # 예) "sentence-transformers/all-MiniLM-L6-v2"
    repo_id: str

    # filename: 리포 안에 있는 파일명
    # 기본은 "README.md"
    filename: str = "README.md"

    # revision: 브랜치/태그/커밋을 지정할 때 사용
    # - None이면 기본 브랜치(보통 main/master)에서 받습니다.
    # - 특정 태그/브랜치에서 받고 싶을 때만 넣으면 됩니다.
    revision: Optional[str] = None

    # max_chars: 문서를 청킹할 때, 한 청크의 최대 길이(문자 수 기준)
    # 문서가 너무 길면 여러 조각으로 잘라 넣어야 검색(RAG)이 잘 됩니다.
    max_chars: int = 1200

    # overlap_chars: 청크끼리 겹치는 문자 수
    # 겹치게 해두면 문맥이 끊기는 문제를 줄일 수 있습니다.
    overlap_chars: int = 150


class IngestPDFRequest(BaseModel):
    source: str = "local_pdf"
    max_chars: int = 1200
    overlap_chars: int = 150

# =========================================================
# 3) /ask 요청 바디(JSON) 형태를 정의
#    질문을 보내고, RAG로 답을 받을 때 사용
# =========================================================
class AskRequest(BaseModel):
    # question: 사용자가 묻는 질문
    # 예) {"question": "Ollama 기본 포트는 뭐야?", "top_k": 3}
    question: str

    # top_k: 검색할 문서(청크) 개수
    # - top_k가 크면 더 많은 문서를 근거로 보지만, 관련 없는 내용도 섞일 수 있음
    # - top_k가 작으면 빠르지만, 근거가 부족할 수 있음
    top_k: int = 10


# =========================================================
# 4) /ask 응답(JSON) 형태를 정의
#    서버가 어떤 형태로 응답해야 하는지 "출력 스키마"를 정합니다.
# =========================================================
class AskResponse(BaseModel):
    # answer: 최종 생성된 답변 텍스트
    answer: str

    # retrieved: 실제로 검색되어 답변 근거로 사용된 문서(청크) 리스트
    # 디버깅/검증용으로 매우 유용합니다.
    # 예) ["문서조각1...", "문서조각2..."]
    retrieved: List[str]