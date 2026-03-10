"""
Configuration for the rotation counter pipeline.
Adjust these values to tune performance and cost.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Video processing ---
TARGET_FPS = 3          # Frames per second to sample from video
FRAME_WIDTH = 512       # Resize frames to this width (maintains aspect ratio)

# --- Contact sheet grid ---
GRID_COLS = 2           # Columns per contact sheet
GRID_ROWS = 4           # Rows per contact sheet (cells = GRID_COLS * GRID_ROWS)
CELL_BORDER = 2         # Pixels of border between cells
LABEL_FONT_SIZE = 14    # Size of cell number labels

# --- VLM API ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

PRIMARY_MODEL = "gpt-4o-mini"
FALLBACK_MODEL = "gemini-2.0-flash"

VLM_TEMPERATURE = 0
VLM_SEED = 42
VLM_TIMEOUT = 30        # seconds
VLM_MAX_RETRIES = 3
VLM_MAX_CONCURRENT = 5  # async semaphore limit

# --- Classification ---
VALID_ORIENTATIONS = {"FRONT", "RIGHT_SIDE", "BACK", "LEFT_SIDE", "UNKNOWN"}
MIN_CONFIDENCE_THRESHOLD = 0.65  # Filter frames below this

# --- Smoothing ---
SMOOTHING_WINDOW = 3    # Majority-vote window size for orientation sequence

# --- Validation ---
MIN_DIRECTION_CONSISTENCY = 0.85  # Fraction of transitions that must match detected direction
MIN_UNIQUE_ORIENTATIONS = 3       # Minimum unique cardinal directions for valid count

# --- Logging ---
LOG_DIR = "logs"
SAVE_LOGS = True

# --- Debug ---
DUMP_SHEETS = True         # Save contact sheet images to DUMP_SHEETS_DIR for inspection
DUMP_SHEETS_DIR = "debug_sheets"
