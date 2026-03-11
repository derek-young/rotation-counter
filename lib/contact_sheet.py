"""
Contact sheet composer: groups frames into labeled grids for batch VLM classification.

Instead of 1 API call per frame, we create grids of GRID_COLS x GRID_ROWS frames
and classify all cells in a single API call. This reduces cost and latency significantly.
"""

from __future__ import annotations
import base64
import io
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFont

from config import GRID_COLS, GRID_ROWS, CELL_BORDER, LABEL_FONT_SIZE
from lib.frame_extractor import VideoFrame


CELLS_PER_SHEET = GRID_COLS * GRID_ROWS


@dataclass
class ContactSheet:
    sheet_index: int        # Which sheet (0-based)
    image_b64: str          # Base64-encoded JPEG of the full grid
    cell_count: int         # Number of actual frames (last sheet may be partial)
    frame_indices: list[int]  # VideoFrame.index for each cell (in cell order)
    timestamps: list[float]   # Timestamp for each cell


def compose_contact_sheets(
    frames: list[VideoFrame],
    cols: int = GRID_COLS,
    rows: int = GRID_ROWS,
) -> list[ContactSheet]:
    """
    Arrange frames into contact sheets for batch VLM classification.

    Each sheet image contains a grid of COLS x ROWS frames, each numbered
    starting at 1. The VLM is asked to return orientations keyed by these
    cell numbers.

    Args:
        frames: Ordered VideoFrame list from frame_extractor.
        cols: Grid columns per sheet.
        rows: Grid rows per sheet.

    Returns:
        List of ContactSheet objects, one per grid image.
    """
    cells_per = cols * rows
    sheets: list[ContactSheet] = []

    for sheet_idx, start in enumerate(range(0, len(frames), cells_per)):
        batch = frames[start : start + cells_per]
        sheet_img = _build_grid(batch, cols, rows)
        b64 = _pil_to_base64(sheet_img)

        sheets.append(
            ContactSheet(
                sheet_index=sheet_idx,
                image_b64=b64,
                cell_count=len(batch),
                frame_indices=[f.index for f in batch],
                timestamps=[f.timestamp for f in batch],
            )
        )

    print(f"[contact_sheet] composed {len(sheets)} sheet(s) from {len(frames)} frames")
    return sheets


def _build_grid(frames: list[VideoFrame], cols: int, rows: int) -> Image.Image:
    """Build a single contact sheet image with numbered cells."""
    if not frames:
        raise ValueError("Cannot build grid from empty frame list")

    # All frames are the same width (FRAME_WIDTH), but height may vary
    cell_w, cell_h = frames[0].image.size

    # Canvas size
    canvas_w = cols * cell_w + (cols + 1) * CELL_BORDER
    canvas_h = rows * cell_h + (rows + 1) * CELL_BORDER

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 30))
    draw = ImageDraw.Draw(canvas)

    # Try to load a font; fall back to default
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", LABEL_FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    for cell_num, frame in enumerate(frames, start=1):
        col = (cell_num - 1) % cols
        row = (cell_num - 1) // cols

        x = CELL_BORDER + col * (cell_w + CELL_BORDER)
        y = CELL_BORDER + row * (cell_h + CELL_BORDER)

        canvas.paste(frame.image, (x, y))

        # Draw cell number label (white text with black shadow for contrast)
        label = str(cell_num)
        draw.text((x + 3, y + 3), label, fill=(0, 0, 0), font=font)      # shadow
        draw.text((x + 2, y + 2), label, fill=(255, 255, 255), font=font) # label

    return canvas


def dump_contact_sheets(sheets: list[ContactSheet], out_dir: str) -> None:
    """Save each contact sheet as a JPEG file for visual inspection."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    for sheet in sheets:
        img_bytes = base64.b64decode(sheet.image_b64)
        img = Image.open(io.BytesIO(img_bytes))
        path = os.path.join(out_dir, f"sheet_{sheet.sheet_index:03d}.jpg")
        img.save(path, format="JPEG")
        print(f"[contact_sheet] dumped {path}")


_ORIENT_COLOR: dict[str, tuple[int, int, int]] = {
    "FRONT":      (80, 200, 80),
    "BACK":       (220, 80, 80),
    "LEFT_SIDE":  (80, 140, 220),
    "RIGHT_SIDE": (220, 180, 50),
    "UNKNOWN":    (160, 160, 160),
}

_ORIENT_SHORT: dict[str, str] = {
    "FRONT":      "FRONT",
    "BACK":       "BACK",
    "LEFT_SIDE":  "LEFT",
    "RIGHT_SIDE": "RIGHT",
    "UNKNOWN":    "?",
}


def dump_labeled_sheets(
    sheets: list[ContactSheet],
    frame_orientations: dict[int, str],
    out_dir: str,
    cols: int = GRID_COLS,
    rows: int = GRID_ROWS,
) -> None:
    """
    Save each contact sheet annotated with per-cell orientation labels.

    Each cell gets its VLM-assigned orientation drawn at the bottom,
    color-coded by orientation type.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", LABEL_FONT_SIZE)
    except Exception:
        font = ImageFont.load_default()

    for sheet in sheets:
        img_bytes = base64.b64decode(sheet.image_b64)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        cell_w = (img.width - (cols + 1) * CELL_BORDER) // cols
        cell_h = (img.height - (rows + 1) * CELL_BORDER) // rows

        for cell_idx, frame_idx in enumerate(sheet.frame_indices):
            orientation = frame_orientations.get(frame_idx, "UNKNOWN")
            label = _ORIENT_SHORT.get(orientation, orientation)
            color = _ORIENT_COLOR.get(orientation, (160, 160, 160))

            col = cell_idx % cols
            row = cell_idx // cols
            x = CELL_BORDER + col * (cell_w + CELL_BORDER)
            y = CELL_BORDER + row * (cell_h + CELL_BORDER)

            # Draw label centered at the bottom of the cell
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            tx = x + (cell_w - text_w) // 2
            ty = y + cell_h - text_h - 4

            draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0), font=font)  # shadow
            draw.text((tx, ty), label, fill=color, font=font)

        path = os.path.join(out_dir, f"sheet_{sheet.sheet_index:03d}.jpg")
        img.save(path, format="JPEG")


def _pil_to_base64(img: Image.Image, quality: int = 85) -> str:
    """Convert PIL Image to base64-encoded JPEG string."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")
