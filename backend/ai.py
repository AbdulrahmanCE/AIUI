import base64
import json
import logging
import os
import time

import openai

AI_COMPLETION_MODEL = os.getenv("AI_COMPLETION_MODEL", "gpt-3.5-turbo")
LANGUAGE = os.getenv("LANGUAGE", "en")
INITIAL_PROMPT = f"""You are Sanad, an AI interviewer assigned to conduct a structured interview with Mazen, a Sales Manager. 

Begin the interview by greeting Mazen and asking him to introduce himself briefly.

Proceed with a series of concise, professional questions, one at a time, focused on:
- Sales strategy and planning
- Client relationship management
- Sales performance indicators (KPIs)
- Collaboration with marketing and product teams
- Handling objections and closing deals
- Use of CRM systems and sales tools
- Adaptability and learning from failure

Keep each question clear and limited to a single sentence, as the interaction is conducted through a voice interface.

Ensure a polite and engaging tone throughout the conversation.

End the interview by thanking Mazen for his time and informing him that he may now leave the meeting.

Always respond in the language that corresponds to the ISO-639-1 code: {LANGUAGE}.
"""


async def get_completion(user_prompt, conversation_thus_far):
    if _is_empty(user_prompt):
        raise ValueError("empty user prompt received")

    start_time = time.time()
    messages = [
        {
            "role": "system",
            "content": INITIAL_PROMPT
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
