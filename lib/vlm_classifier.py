"""
VLM classifier: sends contact sheet grids to an LLM/VLM API and parses
per-cell orientation responses.

Supports OpenAI (primary) and Google Gemini (fallback).
All API calls are async with a semaphore-controlled concurrency limit.
"""

from __future__ import annotations
import asyncio
import json
import re
import time
from openai import AsyncOpenAI, APIStatusError
from typing import Any, TypedDict

import httpx

from config import (
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    PRIMARY_MODEL,
    FALLBACK_MODEL,
    VALID_ORIENTATIONS,
    VLM_TEMPERATURE,
    VLM_SEED,
    VLM_TIMEOUT,
    VLM_MAX_RETRIES,
    VLM_MAX_CONCURRENT,
)
from lib.contact_sheet import ContactSheet

async_client = AsyncOpenAI()


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


SYSTEM_PROMPT_1 = """You are a body orientation classifier analyzing a numbered image grid.

For each numbered cell in the grid, classify the human subject's orientation into EXACTLY ONE of these categories:
- FRONT: Subject facing camera
- RIGHT_SIDE: Subject's right side closest to camera
- BACK: Subject facing away from camera, back of head and shoulders visible
- LEFT_SIDE: Subject's left side closest to camera
- UNKNOWN: No human visible, cell is blank, or orientation cannot be determined

Classification rules:
1. Combine shoulder, hip, arm and head/face planes to get the strongest cue
2. For diagonal orientations (45° between two cardinals), choose the nearest cardinal direction
3. If only partial body is visible, classify based on the visible portion
4. Symmetrical clothing does not affect classification; focus on body geometry

Respond with ONLY a valid JSON object mapping cell number (as string) to orientation.
Example: {"1": "FRONT", "2": "RIGHT_SIDE", "3": "BACK", "4": "LEFT_SIDE"}
No other text, no markdown, no explanation."""


async def classify_sheets(
    sheets: list[ContactSheet],
) -> tuple[dict[int, str], str, int]:
    """
    Classify all contact sheets asynchronously and return a flat
    mapping of frame_index → orientation.

    Args:
        sheets: list of ContactSheets

    Returns:
        Tuple of (frame_orientations, model, total_tokens) where frame_orientations
        maps each frame's VideoFrame.index to its orientation string.
    """
    semaphore = asyncio.Semaphore(VLM_MAX_CONCURRENT)
    results: dict[int, str] = {}
    model_used: str = ""
    total_tokens: int = 0

    async def process_sheet(sheet: ContactSheet) -> None:
        nonlocal model_used, total_tokens
        async with semaphore:
            cell_orientations, sheet_model, sheet_tokens = await _classify_sheet_with_retry(sheet)
        model_used = sheet_model
        total_tokens += sheet_tokens
        # Map cell numbers back to frame indices
        for cell_num_str, orientation in cell_orientations.items():
            cell_num = int(cell_num_str)
            if 1 <= cell_num <= sheet.cell_count:
                frame_idx = sheet.frame_indices[cell_num - 1]
                results[frame_idx] = orientation

    await asyncio.gather(*[process_sheet(s) for s in sheets])

    return results, model_used, total_tokens


async def _classify_sheet_with_retry(sheet: ContactSheet) -> tuple[dict[str, str], str, int]:
    """Classify a single contact sheet, retrying on failure."""

    for attempt in range(VLM_MAX_RETRIES):
        try:
            result = await _call_openai(sheet)
            parsed = _parse_and_validate(result["content"], sheet.cell_count)
            print(
                f"[vlm] sheet {sheet.sheet_index} classified "
                f"({sheet.cell_count} cells, attempt {attempt + 1})"
            )
            return parsed, result["model"], result["tokens"]
        except ValueError:
            # Config/programming errors (e.g. missing API key) won't resolve on retry
            raise
        except APIStatusError as e:
            # Retry on 429 (rate limit) and 5xx (transient server errors);
            # re-raise on other 4xx (bad request, auth, not found, etc.)
            if 400 <= e.status_code < 500 and e.status_code != 429:
                raise
            wait = 2 ** attempt
            print(f"[vlm] sheet {sheet.sheet_index} attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)
        except Exception as e:
            wait = 2 ** attempt
            print(f"[vlm] sheet {sheet.sheet_index} attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)

async def _call_openai(sheet: ContactSheet) -> ModelCallResult:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    
    completion = await async_client.chat.completions.create(
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
                            "url": f"data:image/png;base64,{sheet.image_b64}",
                            "detail": "high",
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

    return { 
        "content": completion.choices[0].message.content,
        "model": completion.model,
        "tokens": completion.usage.total_tokens,
    }


async def _call_gemini(sheet: ContactSheet) -> str:
    """Call Google Gemini API with the contact sheet image."""
    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY not set")

    import google.generativeai as genai

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(FALLBACK_MODEL)

    import base64
    from PIL import Image
    import io

    img_bytes = base64.b64decode(sheet.image_b64)
    pil_img = Image.open(io.BytesIO(img_bytes))

    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Classify the human's body orientation in each of the "
        f"{sheet.cell_count} numbered cells. "
        f"Return JSON with keys '1' through '{sheet.cell_count}'."
    )

    response = await asyncio.to_thread(
        model.generate_content,
        [prompt, pil_img],
        generation_config={"temperature": VLM_TEMPERATURE, "response_mime_type": "application/json"},
    )
    return response.text


def _parse_and_validate(raw: str, expected_cells: int) -> dict[str, str]:
    """
    Parse VLM JSON response and validate orientation values.
    Falls back to regex extraction if JSON is embedded in prose.
    """
    # Try direct JSON parse
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Extract JSON object from prose
        match = re.search(r"\{[^{}]+\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON found in response: {raw[:200]}")
        data = json.loads(match.group())

    # Validate and clean values
    cleaned: dict[str, str] = {}
    for i in range(1, expected_cells + 1):
        key = str(i)
        raw_val = data.get(key, "UNKNOWN").strip().upper()

        if raw_val not in VALID_ORIENTATIONS:
            # Try partial match
            matched = next((v for v in VALID_ORIENTATIONS if v in raw_val), "UNKNOWN")
            raw_val = matched
        cleaned[key] = raw_val

    return cleaned
