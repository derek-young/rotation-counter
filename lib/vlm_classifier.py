"""
VLM classifier: sends contact sheet grids to an LLM/VLM API and parses
per-cell orientation responses.

Supports OpenAI and Google Gemini.
All API calls are async with a semaphore-controlled concurrency limit.
"""

from __future__ import annotations

import asyncio
import json
import re

from openai import APIStatusError

from config import VALID_ORIENTATIONS, VLM_MAX_CONCURRENT, VLM_MAX_RETRIES
from lib.contact_sheet import ContactSheet
from lib.vlm_callers import get_vlm_caller


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
            result = await get_vlm_caller()(sheet)
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
            # Re-raise on non-retriable 4xx (bad request, auth, not found, etc.)
            if 400 <= e.status_code < 500 and e.status_code != 429:
                raise
            wait = 2 ** attempt
            print(f"[vlm] sheet {sheet.sheet_index} attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)
        except Exception as e:
            wait = 2 ** attempt
            print(f"[vlm] sheet {sheet.sheet_index} attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
            await asyncio.sleep(wait)

    raise RuntimeError(f"sheet {sheet.sheet_index} failed after {VLM_MAX_RETRIES} attempts")


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
