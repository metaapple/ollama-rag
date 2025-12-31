# ollama-rag

**Local RAG (Retrieval-Augmented Generation) Application with Ollama**

이 프로젝트는 **Ollama**를 사용하여 로컬에서 실행되는 Retrieval-Augmented Generation (RAG) 시스템입니다. PDF 문서들을 업로드하고, 이를 기반으로 한 지식 베이스를 구축하여 LLM과 대화할 수 있는 웹 인터페이스를 제공합니다.

Ollama의 로컬 LLM (예: llama3, mistral 등)을 활용해 외부 API 없이 완전 오프라인으로 작동하며, PDF 문서에 대한 정확한 질문-답변을 가능하게 합니다.

![Ollama Logo](https://ollama.com/public/ollama.png)
<!-- Ollama 공식 로고 예시 -->

<img width="1378" height="339" alt="스크린샷 2025-12-31 12 14 29" src="https://github.com/user-attachments/assets/e3de7fe4-c21a-48e3-ae76-f07d9c7d9ccb" />

- 관련 ppt : https://docs.google.com/presentation/d/1TA6n-o3OZ52CS_nyF7D4h22mpbq4tmlYrneT1y0BOmM/edit?usp=sharing 


## 특징

- **로컬 실행**: Ollama를 통해 모든 LLM 추론이 로컬에서 수행됩니다. (OpenAI API 불필요)
- **PDF 기반 RAG**: PDF 파일을 업로드하고 자동으로 텍스트 추출 → 청크 분할 → 임베딩 생성 → 벡터 DB 저장
- **웹 UI**: Streamlit 기반의 직관적인 채팅 인터페이스 (스트리밍 응답 지원)
- **스트리밍 지원**: LLM 응답을 실시간으로 스트리밍 출력
- **간단한 구조**: 최소한의 의존성으로 빠르게 실행 가능
- **프라이빗 데이터 처리**: 모든 데이터가 로컬에 저장되어 보안 유지

## 기술 스택

- **Ollama**: 로컬 LLM 실행 및 임베딩 생성
- **LangChain**: RAG 파이프라인 구축 (또는 직접 구현 가능)
- **Streamlit**: 웹 기반 채팅 UI
- **PyPDF2 / pdfplumber**: PDF 텍스트 추출
- **Chroma / FAISS**: 로컬 벡터 데이터베이스 (프로젝트에 따라 다름)
- **Sentence Transformers 또는 Ollama embeddings**: 임베딩 모델

## 사전 요구사항

- **Ollama 설치**: [https://ollama.com/download](https://ollama.com/download)에서 다운로드 및 설치
- Ollama 서비스 실행: 터미널에서 `ollama serve` 실행
- 사용할 LLM 모델 풀링:
  ```bash
  ollama pull llama3  # 또는 mistral, gemma 등 원하는 모델
  ollama pull nomic-embed-text  # 임베딩용 모델 (필요 시)
  ```
- Python 3.10 이상
- Git

## 설치 방법

1. 리포지토리 클론
   ```bash
   git clone https://github.com/metaapple/ollama-rag.git
   cd ollama-rag
   ```

2. 가상 환경 생성 및 활성화 (권장)
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. 의존성 설치
   ```bash
   pip install -r requirements.txt
   ```

   `requirements.txt`에 포함된 주요 패키지 예시:
   ```
   streamlit
   langchain
   langchain-community
   langchain-ollama
   chromadb  # 또는 faiss-cpu
   pypdf
   pdfplumber
   ollama
   ```

## 사용 방법

1. Ollama 서비스 실행 확인
   ```bash
   ollama list  # 설치된 모델 확인
   ```

2. 애플리케이션 실행
   ```bash
   streamlit run app/main.py  # 또는 루트의 메인 파일에 따라 조정 (예: app/app.py)
   ```

3. 웹 브라우저에서 열리는 UI (기본: http://localhost:8501)
   - **PDF 업로드**: `pdf` 폴더에 PDF 파일을 넣거나 UI에서 업로드
   - 문서 인덱싱: "Index Documents" 또는 비슷한 버튼 클릭 → 자동으로 벡터 DB 생성
   - 채팅: 하단 입력창에 PDF 내용 관련 질문을 입력
   - 응답: 관련 문서 청크를 검색해 컨텍스트와 함께 LLM이 답변 생성 (소스 출처 표시 가능)

### 예시 질문
- "이 PDF에서 2024년 매출은 얼마인가?"
- "문서의 주요 결론을 요약해줘."

## 프로젝트 구조

```
ollama-rag/
├── app/              # 메인 애플리케이션 코드 (Streamlit 앱, RAG 로직)
├── pdf/              # 업로드된 PDF 파일 저장 디렉토리
├── stream/           # 스트리밍 관련 코드 또는 캐시 (추정)
├── requirements.txt  # Python 의존성 목록
├── .idea/            # PyCharm IDE 설정 (무시 가능)
└── README.md         # 이 파일
```

- 벡터 DB는 실행 시 로컬에 자동 생성 (예: chromadb 폴더)

## troubleshooting

- **Ollama 연결 오류**: Ollama가 실행 중인지 확인 (`curl http://localhost:11434`)
- **임베딩 모델 필요**: 일부 설정에서 `nomic-embed-text` 모델이 필요할 수 있음
- **메모리 부족**: 큰 PDF나 모델 사용 시 GPU가 있는 환경 권장 (Ollama가 CUDA 지원 시)
- **PDF 추출 오류**: 복잡한 PDF는 pdfplumber가 더 잘 작동할 수 있음

## 기여하기

버그 리포트, 기능 제안, 풀 리퀘스트 환영합니다!

1. Fork the repository
2. 새 브랜치 생성 (`git checkout -b feature/amazing-feature`)
3. 변경 커밋 (`git commit -m 'Add amazing feature'`)
4. 푸시 (`git push origin feature/amazing-feature`)
5. Pull Request 생성

## 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일 참조 (없으면 추가 추천)

## 감사 인사

- [Ollama](https://ollama.com) 팀
- LangChain 및 Streamlit 커뮤니티
- 다양한 오픈소스 RAG 프로젝트에서 영감 얻음

---

문의사항은 Issues에 남겨주세요! 즐거운 로컬 RAG 생활 되세요 🚀
