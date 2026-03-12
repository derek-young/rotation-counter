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
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

PRIMARY_MODEL = "gpt-5.4"
# PRIMARY_MODEL = "gemini-3-flash-preview"

VLM_TEMPERATURE = 0
VLM_SEED = 42
VLM_MAX_RETRIES = 3
VLM_MAX_CONCURRENT = 5  # async semaphore limit

# --- Classification ---
VALID_ORIENTATIONS = {"FRONT", "RIGHT_SIDE", "BACK", "LEFT_SIDE", "UNKNOWN"}

# --- Logging ---
LOG_DIR = "logs"
SAVE_LOGS = True
PERFECT_RUN_PATH = "logs/perfect_run_4x2_3fps.json"
ACCURACY_MOVING_AVG_WINDOW = 10

# --- Debug ---
DUMP_SHEETS = False         # Save contact sheet images to DUMP_SHEETS_DIR for inspection
DUMP_SHEETS_DIR = "debug_sheets"
