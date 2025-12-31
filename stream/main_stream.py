from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import json
import uvicorn

app = FastAPI(title="Gemma3:1b AI Agent with Ollama - Streaming")


class MessageRequest(BaseModel):
    prompt: str  # 사용자 입력 메시지


# Ollama 스트리밍 응답을 그대로 전달하기 위한 제너레이터
def ollama_stream(prompt: str):
    ollama_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma3:1b",
        "prompt": prompt,
        "stream": True  # 스트리밍 활성화
    }

    try:
        with requests.post(ollama_url, json=payload, stream=True) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    # Ollama는 각 줄이 JSON 객체
                    chunk = json.loads(line.decode("utf-8"))
                    if "response" in chunk:
                        # 각 chunk의 텍스트 부분만 yield (SSE 형식으로)
                        yield f"data: {json.dumps({'response': chunk['response']})}\n\n"
                    if chunk.get("done", False):
                        # 완료 시 done 신호 전달 후 종료
                        yield f"data: {json.dumps({'done': True})}\n\n"
                        break
    except requests.exceptions.RequestException as e:
        error_msg = {"error": "Ollama 서버와 통신 실패", "details": str(e)}
        yield f"data: {json.dumps(error_msg)}\n\n"


@app.post("/chat")
async def chat_stream(request: MessageRequest):
    # Server-Sent Events (SSE) 형식으로 스트리밍
    return StreamingResponse(
        ollama_stream(request.prompt),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)