# mysql연결하는 것처럼 포트 11434인 올라마서버에 연결하는 모듈
from fastapi import HTTPException
import requests

# http서버이므로 http연결하는 모듈 필요함.
# 순서대로 호출해서 받을 것이면 requests(*), tablib.request
# 동시에 호출해서 받을 것이면 httpx
# Ollama 기본 설정
OLLAMA_BASE_URL = "http://localhost:11434"  # Ollama 기본 포트
# DEFAULT_MODEL = "llama3.2"
DEFAULT_MODEL = "gemma3:1b"

def ollama_client(word : str):
    # word가 없으면 올라마 서버 호출할 필요없음.
    if not word or word.strip() == "":
        raise HTTPException(status_code=400, detail="word값이 없다. 프롬프트에 값을 넣으시오.!")
    #외부 자원과의 연결은 위험한 순간이므로 try-except를 해줌.
    import httpx
    try :
        # 올라마 서버로 보낼 데이터 만들고
        payload = {
            "model": DEFAULT_MODEL,
            "prompt": word.strip(),
            "stream": False,  # 스트리밍 비활성화로 전체 응답 받기
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 100
            }
        }
        # 올라마 서버로 post요청을 보내서
        response = requests.post(url=f"{OLLAMA_BASE_URL}/api/generate",
                                 json=payload)
        # 결과 받아오면 결과를 추출해서
        print(response.status_code)
        print(response)
        if response.status_code == 200:
            ollama_response = response.json()
            print(ollama_response)
        # 우리마음대로 결과 dict을 만들어주자.
            # JSON 응답 구성
            result = {
                "success": True,
                "question": word,
                "answer": ollama_response.get("response", ""),
                "model": ollama_response.get("model", DEFAULT_MODEL),
                "metadata": {
                    "total_duration": ollama_response.get("total_duration"),
                    "load_duration": ollama_response.get("load_duration"),
                    "prompt_eval_count": ollama_response.get("prompt_eval_count"),
                    "eval_count": ollama_response.get("eval_count"),
                    "eval_duration": ollama_response.get("eval_duration")
                }
            }
            return result
    except httpx.ConnectError:
        print("연결 에러")
    except httpx.HTTPError:
        print("통신 에러")
    except httpx.TimeoutException:
        print("타임 아웃 에러")
    except Exception as e:
        print("올라마 서버와의 통신중에 에러가 발생함. " + str(e))
        return {'success': False}