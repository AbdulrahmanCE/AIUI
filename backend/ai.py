import base64
import json
import logging
import os
import time
import httpx  # Add this import

import openai

AI_COMPLETION_MODEL = os.getenv("AI_COMPLETION_MODEL", "gpt-3.5-turbo")
LANGUAGE = os.getenv("LANGUAGE", "en")
PROMPT_API_URL = "https://portal.solvfast.com/api/method/interview_prompt"

_prompt_cache = None

async def fetch_initial_prompt():
    global _prompt_cache
    if _prompt_cache is not None:
        return _prompt_cache
    async with httpx.AsyncClient() as client:
        resp = await client.get(PROMPT_API_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        prompt = data.get("message", "")
        # Optionally, you can format with LANGUAGE if needed
        _prompt_cache = prompt.replace("{LANGUAGE}", LANGUAGE)
        return _prompt_cache

async def get_completion(user_prompt, conversation_thus_far):
    if _is_empty(user_prompt):
        raise ValueError("empty user prompt received")

    start_time = time.time()
    initial_prompt = await fetch_initial_prompt()
    messages = [
        {
            "role": "system",
            "content": initial_prompt
        }
    ]

    messages.extend(json.loads(base64.b64decode(conversation_thus_far)))
    messages.append({"role": "user", "content": user_prompt})

    logging.debug("calling %s", AI_COMPLETION_MODEL)
    res = await openai.ChatCompletion.acreate(model=AI_COMPLETION_MODEL, messages=messages, timeout=15)
    logging.info("response received from %s %s %s %s", AI_COMPLETION_MODEL, "in", time.time() - start_time, "seconds")

    completion = res['choices'][0]['message']['content']
    logging.info('%s %s %s', AI_COMPLETION_MODEL, "response:", completion)

    return completion


def _is_empty(user_prompt: str):
    return not user_prompt or user_prompt.isspace()
