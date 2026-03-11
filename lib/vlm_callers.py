"""
VLM caller implementations: one function per model provider.

Each caller accepts a ContactSheet and returns a ModelCallResult.
Register new callers in _MODEL_DISPATCH to make them available.
"""

from __future__ import annotations

import base64
import time
from google import genai
from google.genai import types
from openai import AsyncOpenAI
from typing import Awaitable, Callable, TypedDict

from config import GOOGLE_API_KEY, OPENAI_API_KEY, PRIMARY_MODEL, VLM_SEED, VLM_TEMPERATURE
from lib.contact_sheet import ContactSheet

async_openai_client = AsyncOpenAI()
gemini_client = genai.Client(api_key=GOOGLE_API_KEY)


class ModelCallResult(TypedDict):
    content: str
    model: str
    tokens: int


SYSTEM_PROMPT = """
You are an expert in human pose estimation and spatial orientation.

Your task is to analyze a numbered image grid and determine the precise direction a person is facing in each quadrant.

For each numbered cell in the grid, classify the human subject's orientation into EXACTLY ONE of these categories:
- FRONT
- RIGHT_SIDE
- BACK
- LEFT_SIDE
- UNKNOWN

# Methodology
Before providing a final orientation, you MUST perform a structured "Anatomical Audit" by answering the following questions:

1. **Head & Face:** Are both eyes visible? Is the bridge of the nose centered or on the silhouette's edge? Is the back of the head (hair/occipital) the primary texture?
2. **Torso & Shoulders:** Are the shoulders "stacked" (one obscuring the other) or "parallel" to the lens? Are chest features (sternum, buttons, tie) or back features (scapula, spine line) visible?
3. **Arms:** Are both arms equidistant to the center point of the camera? Is the subject's left arm or right arm closer to the camera?
4. **Hips:** Are the hips "stacked" or "parallel" to the lens? Is the right hip or left hip closer to the camera?

**Clock Mapping:** Place the person at the center of a clock face. The camera is fixed at the 6:00 position. Determine where the person is facing on this clock:
- 6:00 = **FRONT** (facing camera; both eyes visible, chest features visible)
- 12:00 = **BACK** (facing away; back of head visible, scapulae/spine line visible)
- 3:00 = **RIGHT_SIDE** (camera sees the person's right profile)
- 9:00 = **LEFT_SIDE** (camera sees the person's left profile)

For diagonal angles (e.g., 4:30), choose the nearest cardinal direction. Use your anatomical audit to triangulate: face, shoulders, and hips should all agree on a clock position — note any that disagree and weight the majority.

# Reasoning Process
Work through the four anatomical audit questions silently. You do not output this reasoning — it exists only to inform your final answer. Resolve any conflicts between cues (e.g., face says 6:00 but shoulders say 4:30) by choosing the nearest cardinal direction.

# Output Format
Respond with ONLY a valid JSON object mapping cell number (as string) to final orientation.
Example: {"1": "FRONT", "2": "RIGHT_SIDE", "3": "BACK", "4": "LEFT_SIDE"}
No other text, no markdown, no explanation.
"""

async def _call_gemini(sheet: ContactSheet) -> ModelCallResult:
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set")

    start_time = time.time()

    # Gemini expects raw bytes for image parts
    image_bytes = base64.b64decode(sheet.image_b64)
    
    response = await gemini_client.aio.models.generate_content(
        model=PRIMARY_MODEL, 
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            (
                f"Classify the human's body orientation in each of the "
                f"{sheet.cell_count} numbered cells. "
                f"Return JSON with keys '1' through '{sheet.cell_count}'."
            ),
        ],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=VLM_TEMPERATURE,
            response_mime_type="application/json",
        ),
    )

    server_elapsed = time.time() - start_time
    usage = response.usage_metadata

    print(f"[vlm] sheet {sheet.sheet_index} server elapsed: {server_elapsed:.2f}s")
    print(f"[vlm] prompt tokens: {usage.prompt_token_count}, candidates tokens: {usage.candidates_token_count}")

    return {
        "content": response.text,
        "model": response.model_version,
        "tokens": usage.total_token_count,
    }


async def _call_openai(sheet: ContactSheet) -> ModelCallResult:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")

    completion = await async_openai_client.chat.completions.create(
        model=PRIMARY_MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{sheet.image_b64}",
                            "detail": "low",
                        }
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Classify the human's body orientation in each of the "
                            f"{sheet.cell_count} numbered cells. "
                            f"Return JSON with keys '1' through '{sheet.cell_count}'."
                        ),
                    },
                ],
            },
        ],
        response_format={"type": "json_object"},
        seed=VLM_SEED,
        temperature=VLM_TEMPERATURE,
    )

    server_elapsed = time.time() - completion.created
    print(f"[vlm] sheet {sheet.sheet_index} server elapsed: {server_elapsed:.2f}s")
    print(f"[vlm] sheet {sheet.sheet_index} input tokens: {completion.usage.prompt_tokens}, output tokens: {completion.usage.completion_tokens}")

    return {
        "content": completion.choices[0].message.content,
        "model": completion.model,
        "tokens": completion.usage.total_tokens,
    }


_MODEL_DISPATCH: dict[str, Callable[[ContactSheet], Awaitable[ModelCallResult]]] = {
    "gemini": _call_gemini,
    "gpt": _call_openai,
}


def get_vlm_caller() -> Callable[[ContactSheet], Awaitable[ModelCallResult]]:
    for prefix, fn in _MODEL_DISPATCH.items():
        if PRIMARY_MODEL.startswith(prefix):
            return fn
    raise ValueError(f"No handler registered for model: {PRIMARY_MODEL!r}")
