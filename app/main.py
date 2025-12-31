import ollama
import asyncio
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
# from ollama_client import stream_generate
from redis.asyncio import Redis  # redis-py의 async 클라이언트
import json

from chroma_db import ChromaRAG
from fastapi import UploadFile, File

# from fastapi.middleware.cors import CORSMiddleware
# from transformers import pipeline


class HealthResponse(BaseModel):
    status: str
    ollama_status: str
    message: str


app = FastAPI()

# Redis 설정 (로컬 개발 기준, 프로덕션에서는 환경변수로 관리)
REDIS_URL = "redis://localhost:6379"
# 앱 상태에 Redis 클라이언트 저장
app.state.redis = None  # 아직 연결안됨. fastapi시작할 때 redis도 연결해두려고 함.

# Static 파일 설정 (CSS, JS, 이미지 등)
app.mount("/static", StaticFiles(directory="static"), name="static")
# Templates 설정
templates = Jinja2Templates(directory="templates")

# CORS 설정 (클라이언트 도메인 허용, 개발 시 "*"로 테스트)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # 프로덕션에서는 특정 도메인으로 제한, 예: ["http://localhost:3000"]
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# 사용할 모델 이름 (미리 ollama pull <model>로 다운로드 필요, 예: ollama pull llama3.2)
# MODEL = "llama3.2:3b"
# MODEL = "gemma3:4b"
MODEL = "gemma3:1b"
OLLAMA_BASE_URL = "http://localhost:11434"

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})

@app.get("/ollama-test")
def ollamatest(request: Request):
    return templates.TemplateResponse("ollama-test.html", context={"request": request})


@app.get("/health")
async def health_check():
    """FastAPI와 Ollama의 health 상태를 확인하는 엔드포인트"""
    try:
        # Ollama health check
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)

        if response.status_code == 200:
            ollama_status = "healthy"
            message = "fastapi & ollama 제대로 동작중"
        else:
            ollama_status = "unhealthy"
            message = f"Ollama returned status code: {response.status_code}"

    except httpx.ConnectError:
        ollama_status = "연결불가"
        message = "Ollama 연결할 수 없음."
    except httpx.TimeoutException:
        ollama_status = "타임아웃"
        message = "Ollama 타임 아웃"
    except Exception as e:
        ollama_status = "error"
        message = f"Error checking Ollama: {str(e)}"

    return HealthResponse(
        status="ok",
        ollama_status=ollama_status,
        message=message
    )


# 앱 시작 시 모델 미리 로드 (preload)
@app.on_event("startup")
async def preload_model():
    try:
        # 빈 프롬프트로 모델 로드 + 영구 유지
        await ollama.AsyncClient().generate(
            model=MODEL,
            prompt=" ",  # 빈 프롬프트 (또는 "preload" 같은 더미 텍스트)
            keep_alive=-1  # -1: 영구적으로 메모리에 유지
        )
        print(f"{MODEL} 모델이 미리 로드되었습니다. (메모리에 영구 유지)")

        app.state.redis = Redis.from_url(url=REDIS_URL, decode_responses=True)
        # decode_responses=True --> 바이트스트림으로 도착한 데이터 utf-8로 자동으로 변환

        print(f"{REDIS_URL}로 Redis서버 미리 연결됨.")

    except Exception as e:
        print(f"모델 preload 실패 또는 Redis연결 실패 : {e}")


# fastapi서버가 종료(재부팅)되었을 때 자동 호출됨.
@app.on_event("shutdown")
async def shutdown_event():
    if app.state.redis:
        await app.state.redis.close()
        print("redis 연결 종료됨.....")


# 일반 generate 엔드포인트 (스트리밍 없이 전체 응답)
@app.get("/chat")
async def generate(word: str, request: Request):
    try:
        response = await ollama.AsyncClient().generate(
            model=MODEL,
            prompt=word,
            options={"temperature": 1},
            keep_alive=-1  # 필요 시 후속 요청에서도 유지
        )
        return templates.TemplateResponse("chat.html",
                                          context={"request": request,
                                                   "result": response["response"]
                                                   })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 스트리밍 엔드포인트 (실시간 토큰 반환, 더 빠른 체감)
from fastapi.responses import StreamingResponse


@app.get("/stream")
async def stream(word: str):
    return StreamingResponse(stream_generate(word), media_type="text/event-stream")


async def stream_generate(prompt: str):
    stream = await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        stream=True,
        keep_alive=-1
    )
    async for part in stream:
        # "이 값을 내보내고, 여기서 잠깐 멈춰. 다음에 다시 불러주면 이어서 할게!"
        # ollama로 부터 받은 조각마다 보내..
        yield part["response"]


@app.get("/ollama-rag2")
def ollama_test(request: Request):
    return templates.TemplateResponse("ollama-rag2.html", context={"request": request})


## 파라메터 전달용 class를 만들자.
## 파라메터 이름 똑같은거 자동으로 변수에 들어감.
## 다른 옵션값들 설정 가능
## BaseModel이라는 클래스를 상속받아서 만들어야 자동으로
## 이런 처리들을 해줌.

class SummarizeRequest(BaseModel):
    ## BaseModel(변수+함수) + 내가 추가한 변수
    text: str
    max_length: int = 200


@app.post("/summarize")
async def summarize(request: SummarizeRequest):
    # http://localhost:11434/api/generate, json=payload
    # post방식으로 http요청을 해줌.
    prompt = f"{request.text}를 {request.max_length}자로 요약해주세요."
    print(prompt)
    response = await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )
    print("-----------------")
    print(response)  # dict
    return {'summary': response["response"].strip()}


class TranslateRequest(BaseModel):
    text: str


@app.post("/translate")
async def translate(request: TranslateRequest):
    prompt = f"다음 영어 문장을 자연스러운 한국어로 번역해 주세요. 번역만 출력하세요:\n\n{request.text}"
    response = await ollama.AsyncClient().generate(model=MODEL, prompt=prompt, keep_alive=-1)
    return {"translation": response["response"].strip()}


class SentimentRequest(BaseModel):
    text: str


@app.post("/sentiment")
async def sentiment(request: SentimentRequest):
    prompt = f"""
    다음 문장의 감정을 분석해 주세요. 답변은 반드시 다음 중 하나만 출력하세요: 긍정, 부정, 중립

    문장: {request.text}
    """
    response = await ollama.AsyncClient().generate(model=MODEL, prompt=prompt, keep_alive=-1)
    sentiment = response["response"].strip()
    return {"sentiment": sentiment}


class BrainstormRequest(BaseModel):
    topic: str
    count: int = 5


@app.post("/brainstorm")
async def brainstorm(request: BrainstormRequest):
    prompt = f"""
    주제 '{request.topic}'에 대해 창의적이고 실현 가능한 아이디어를 {request.count}개 제안해 주세요.
    각 아이디어는 번호를 붙이고 한 문장으로 간단히 설명하세요.
    """
    response = await ollama.AsyncClient().generate(model=MODEL, prompt=prompt, keep_alive=-1)
    return {"ideas": response["response"].strip()}


# 6. 시 쓰기 도우미
class PoemRequest(BaseModel):
    topic: str
    style: str = "현대시"  # 예: 현대시, 전통 시조, 자유시 등


@app.post("/poem")
async def write_poem(request: PoemRequest):
    prompt = f"""
    다음 주제로 한국어로 아름다운 시를 한 편 지어주세요.
    스타일은 '{request.style}'로 해주세요. 감성적이고 운율이 살아있게 해주세요.

    주제: {request.topic}

    제목도 함께 붙여주세요.
    """
    response = await ollama.AsyncClient().generate(model=MODEL, prompt=prompt, keep_alive=-1)
    return {"poem": response["response"].strip()}


# 7. 레시피 생성기
class RecipeRequest(BaseModel):
    ingredients: str  # 예: "계란, 토마토, 양파, 치즈"
    servings: int = 2
    difficulty: str = "쉬움"  # 쉬움, 보통, 어려움


@app.post("/recipe")
async def generate_recipe(request: RecipeRequest):
    prompt = f"""
    다음 재료를 사용해서 {request.servings}인분 요리를 만들어 주세요.
    난이도는 '{request.difficulty}' 수준으로, 단계별로 자세히 설명해 주세요.

    재료: {request.ingredients}

    요리 이름도 창의적으로 지어주고, 필요한 추가 재료(조미료 등)는 최소한으로 제안해 주세요.
    """
    response = await ollama.AsyncClient().generate(model=MODEL, prompt=prompt, keep_alive=-1)
    return {"recipe": response["response"].strip()}


class NameRequest(BaseModel):
    # axios.post로 전달될 때 키와 이름이 같아야한다.
    # {category : "아기", gender : "여성", ....}
    category: str = "카페"
    gender: str = "중성"
    count: int = 3
    vibe: str = "따뜻한"


@app.post("/names")
async def names(request: NameRequest):
    prompt = f"""
                {request.category}이름을 
                {request.gender}, {request.vibe}느낌으로 
                {request.count}개만 추천해줘.

            결과 화면은 다음과 같이 만들어줘.

            1. 이름 - 간단설명
            2. 이름 - 간단설명
            3. 이름 - 간단설명
            """
    print(prompt)
    response = await ollama.AsyncClient().generate(
        model=MODEL,
        prompt=prompt,
        keep_alive=-1
    )

    print("-----------------")
    print(response)  # dict
    return {'names': response["response"].strip()}


# # 요약 파이프라인 (한 번만 로드)
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")


# def extract_text_from_url(url: str) -> str:
#     response = requests.get(url)
#     soup = BeautifulSoup(response.text, 'html.parser')
#
#     # 간단한 추출 (사이트별로 다름, 더 정확하게 하려면 readability 사용)
#     for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
#         script.decompose()
#
#     text = soup.get_text(separator='\n')
#     lines = (line.strip() for line in text.splitlines())
#     chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
#     text = '\n'.join(chunk for chunk in chunks if chunk)
#     return text

# class TextRequest(BaseModel):
#     text: str

# @app.post("/summarize-text")
# async def summarize_text(request: TextRequest):
#     # 예: transformers로 요약
#     summary = summarizer(request.text[:1000], max_length=300, min_length=50, do_sample=False)[0]['summary_text']
#     return {"summary": summary}

# 추가 엔드포인트
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"  # 간단히 세션 구분


# 메모리 기반 간단한 대화 히스토리 저장 (프로덕션에서는 Redis 등 사용)
chat_histories = {}


########### redis연결전
# @app.post("/chat")
# async def chat(request: ChatRequest):
#     history = chat_histories.get(request.session_id, [])
#     history.append({"role": "user", "content": request.message + ", 200글자 이내로 핵심만 답을 줘."})
#     # 나는 user, ai는 assistant
#
#     response = await ollama.AsyncClient().chat(
#         model=MODEL,
#         messages=history,
#         keep_alive=-1
#     )
#
#     print("-----------------")
#     print(response)
#     # message = Message(role='assistant',
#     #                   content='싱가포르는 매력적인 도시로, 다양한 경험을 제공하는 매혹적인 나라입니다. 싱가포르 여행에 대한 유용한 정보들을 정리해 드릴게요.\n\n**1. 여행 준비**\n\n*   **비자:** 한국인은 90일까지 무비자 체류 가능합니다
#     ai_message = response["message"]["content"]
#     history.append({"role": "assistant", "content": ai_message})
#     chat_histories[request.session_id] = history[-10:]  # 최근 10턴 유지
#
#     print("chat_histories>> ", chat_histories)
#     # chat_histories >> {'default': [{'role': 'user', 'content': '싱가폴 여행정보, 200글자 이내로 핵심만 답을 줘.'},
#     # {'role': 'assistant', 'content': '싱가포르는 다채로운 문화와 현대적인 도시, 그리고 맛있는 음식으로 유명합니다. \n\n*   **교통:** 대중교통 시스템이 매우 잘 갖춰져 있어 편리하게 이동할 수 있습니다.\n*   **관광 명소:** 마리나 베이 푹, 센토사 섬, 칠리 궁전 등 다양한 명소가 있습니다.\n*   **특징:** 24시간 영업하는 식료품점, 다양한 길거리 음식, 럭셔리한 쇼핑 등 독특한 경험을 할 수 있습니다.\n\n**여행 준비:** 미리 항공권과 숙소를 예약하고, 환전은 한국 원화로 하는 것이 좋습니다.'}]}
#     return {"response": ai_message}


# redis에 키를 chat_history:로그인id로 만들어줄 예정.
# id가 apple인 경우 키는 chat_history:apple
# id가 default인 경우 키는 chat_history:default
# chat_history:apple, chat_history:default는 키이므로 unique해야함.
class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"  # 로그인한 아이디


# 기존 /chat 엔드포인트 수정
########### redis연결후
@app.post("/chat")
async def chat(request: ChatRequest):
    print(f"서버로 전달된 값은 {request.message}, {request.session_id}")
    history = []
    # prompt를 user_message에 만들어주세요.
    user_message = f"{request.message}를 200자 이내로 답변을 줘라. 단답형으로 줘라. 응답은 리스트 형태로 줘라."
    # ollama.AsyncClient().chat()쓸때는
    # - 내가 쓴 것은 role:user가 되어야만 함.
    # - 응답받은 것은 role:assistant가 됨.
    # ollama에게 질문을 줄때는 [{}]로 주어야함.

    history.append({"role": "user", "content": request.message})

    # ollama연결해서 응답받고, 리턴
    response = await ollama.AsyncClient().chat(
        model=MODEL,
        messages=history,
        keep_alive=-1
    )

    print("-----------------")
    print(response)

    # 올라마의 결과는 dict로 온다.
    # response변수에 저장함.--> {message : {content : 응답내용}}
    ai_message = response["message"]["content"]
    history.append({"role": "assistant", "content": ai_message})

    print("chat_histories>> ", history)

    ## redis에 넣자.!
    session_key = "chat_history:" + request.session_id
    # history가 있으면 넣자.!!

    redis = app.state.redis

    if history:
        # redis에는 json으로 넣어주어야한다.
        # 우리는 dict를 가지고 있다 --> json으로 바꾸어주어야함.
        # json.dumps(dict)
        await redis.rpush(session_key, *[json.dumps(one) for one in history])

    return {"response": response['message']['content']}


@app.get("/chat-history/{session_id}")
async def chat_history(session_id: str):
    # 1. 레디스가 연결이 안되어있으면 500번에러
    redis = app.state.redis
    if not redis:  # redis가 None이면
        raise HTTPException(status_code=500, detail="Redis 연결 안됨.")
        # http응답을 보내버림(code, detail을 http 헤더에 넣어서 브라우저에 응답함.)
        # http만들어서 응답하고 끝!

    # 2. 레이스가 연결이 되어있으면
    session_key = 'chat_history:' + session_id
    # [1,2,2,34,354,5435345,35,3,45,353,5,35,35,353,5,3,5,3,4,5,6]
    # 6은 -1인덱스임.

    #    redis.lrange() 리스트를 불러오자.
    history_json = await redis.lrange(session_key, 0, -1)
    print("==========================================")
    print(history_json)  ## [ '{}', '{}', ...] ==> [{}, {}, {},....]
    # for문 돌려서 '{}'이렇게 생긴 string을 빼서 json으로 바꿔서,
    # json의 리스트로 만들어주어야함.
    history = [json.loads(msg) for msg in history_json]
    # [{}, {}, {}, ....]
    print("*********************************")
    print(history)
    return {"history": history}

##################################
# 크로마db test
##################################


from fastapi import HTTPException
from schemas import *
from chroma_db import ChromaRAG
from fastapi import UploadFile, File


# RAG 엔진(전역 1개)
rag = ChromaRAG(
    chroma_dir="./chroma_data",
    collection_name="rag_docs",
    ollama_base_url="http://localhost:11434",
    embed_model="nomic-embed-text",
    gen_model="gemma3:1b",
)


# 텍스트를 읽어서 청크--> 임베딩 --> 크로마db에 넣는 요청
@app.post("/ingest_texts")
def ingest_texts(req: IngestTextsRequest):
    if not req.texts:
        raise HTTPException(status_code=400, detail="texts is empty")

    added = rag.ingest_texts(req.texts, source=req.source)
    return {"docs_added": added, "total_docs": rag.count()}

# pdf파일을 업로드해서 텍스트로 변환한 후,
# 텍스트를 읽어서 청크--> 임베딩 --> 크로마db에 넣는 요청
@app.post("/ingest_pdf")
async def ingest_pdf(
        file: UploadFile = File(...),
        max_chars: int = 1200,
        overlap_chars: int = 150,
        source: str = "pdf",
):
    """
    PDF 파일을 업로드 받아 텍스트 추출 → 청킹 → Chroma 저장
    - 파라미터는 query string 형태로도 받을 수 있게 최소로 구성
    - 예: /ingest_pdf?max_chars=1200&overlap_chars=150&source=pdf
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are allowed")

    ## 파일을 읽은 이후
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # PDF → 텍스트 추출
    text = rag.pdf_to_text(pdf_bytes)
    if not text.strip():
        # 스캔 PDF(이미지)면 텍스트가 없을 수 있음
        raise HTTPException(
            status_code=400,
            detail="No extractable text found. If it's a scanned PDF, OCR is needed."
        )

    # 청킹 후 임베딩 후 저장
    chunks_added = rag.ingest_document(
        raw_text=text,
        source=source,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        meta_extra={"filename": file.filename},
    )

    return {"chunks_added": chunks_added, "total_docs": rag.count(), "filename": file.filename}


@app.post("/ask")
def ask(req: AskRequest):
    # 문서가 하나도 없으면 질문해도 의미가 없으니 400 처리
    if rag.count() == 0:
        raise HTTPException(status_code=400, detail="No documents. Ingest first.")

    # RAG 실행 (크로마 db에서 검색 -> 안되면 올라마 생성)
    out = rag.ask(req.question, top_k=req.top_k)

    print("====================")
    print(out)
    # dict로 그대로 반환하면 FastAPI가 JSON으로 바꿔서 응답함
    # out 구조: {"answer": "...", "retrieved": ["...", "..."]}
    return out
