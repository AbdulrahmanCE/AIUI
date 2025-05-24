import base64
import json
import time
import logging
import threading
from datetime import datetime

import requests
from fastapi import FastAPI, UploadFile, BackgroundTasks, Header
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ai import get_completion
from stt import transcribe
from tts import to_speech

app = FastAPI()
logging.basicConfig(level=logging.INFO)

WEBHOOK_URL = "https://autoflow.solvfast.com/webhook/a08141a7-5256-4ed6-b5a9-8adea15803b9"

@app.post("/inference")
async def infer(audio: UploadFile, background_tasks: BackgroundTasks,
                conversation: str = Header(default=None)) -> FileResponse:
    logging.debug("received request")
    start_time = time.time()

    user_prompt_text = await transcribe(audio)
    ai_response_text = await get_completion(user_prompt_text, conversation)
    ai_response_audio_filepath = await to_speech(ai_response_text, background_tasks)

    # Log user and AI messages asynchronously
    _log_message("applicant", user_prompt_text)
    _log_message("ai", ai_response_text)

    logging.info('total processing time: %s %s', time.time() - start_time, 'seconds')
    return FileResponse(path=ai_response_audio_filepath, media_type="audio/mpeg",
                        headers={"text": _construct_response_header(user_prompt_text, ai_response_text)})


@app.get("/")
async def root():
    return RedirectResponse(url="/index.html")


app.mount("/", StaticFiles(directory="/app/frontend/dist"), name="static")


def _construct_response_header(user_prompt, ai_response):
    return base64.b64encode(
        json.dumps(
            [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": ai_response}]
        ).encode('utf-8')
    ).decode("utf-8")


def _log_message(source: str, text: str):
    def send():
        try:
            requests.post(
                WEBHOOK_URL,
                json={
                    "timestamp": datetime.utcnow().isoformat(),
                    "source": source,
                    "text": text
                },
                timeout=2
            )
        except Exception as e:
            logging.warning(f"Failed to log message: {e}")

    threading.Thread(target=send, daemon=True).start()
