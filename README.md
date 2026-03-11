# LLM-Based Rotation Counter

Counts the number of full 360° human body rotations in a video using only LLMs/VLMs — no traditional CV tracking models.

```
python main.py rotationsTest.mp4
# → 5
```

---

## Architecture

```
Video → Frame Extraction (OpenCV) → Contact Sheet Grids → VLM Classification (async)
     → Orientation Sequence → Smoother → State Machine Counter → Validator → int
```

### The Core Insight: Contact Sheet Batching

Instead of one API call per frame (expensive, slow), frames are arranged into numbered grid images (contact sheets). A single API call classifies all cells at once.

At 3 FPS for a 30-second video: 90 frames → 12 contact sheets (2×4 grids) → **12 API calls** instead of 90.

This provides ~7× cost and speed improvement with no accuracy loss.

### Why This Avoids Temporal Tracking Failures

LLMs fail at continuous temporal tracking because they cannot maintain internal state across frames. This pipeline sidesteps that by:

1. **Reducing the LLM's job to a single classification** (what direction is this person facing?) — not "count the rotations"
2. **Handling all temporal reasoning algorithmically** in the deterministic state machine
3. **Treating the LLM output as noisy sensor data** and smoothing it before counting

---

## Pipeline Components

| File | Responsibility |
|------|---------------|
| `frame_extractor.py` | Sample frames from video at configurable FPS using OpenCV |
| `contact_sheet.py` | Compose numbered frame grids (PIL/Pillow) for batch classification |
| `vlm_classifier.py` | Async VLM API calls (OpenAI primary, Gemini fallback) with retry logic |
| `rotation_algorithm.py` | Orientation smoothing, direction detection, cumulative angle counting |
| `validator.py` | Sanity checks and confidence scoring |
| `main.py` | Orchestrates the pipeline; outputs a single integer |
| `config.py` | All tunable parameters in one place |

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Set API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Run
python main.py rotationsTest.mp4

# Run 5 consecutive times (consistency test)
chmod +x scripts/run_5x.sh
./scripts/run_5x.sh rotationsTest.mp4

# Unit tests
python -m pytest tests/ -v
```

---

## Prompting Strategy

The VLM receives a contact sheet (grid of numbered frames) with this system prompt structure:

- **Constrained output**: Only 4 valid orientation labels (FRONT, RIGHT_SIDE, BACK, LEFT_SIDE, UNKNOWN)
- **Disambiguation rules**: "Use shoulder/hip plane, ignore head/face/arms"
- **Diagonal handling**: Explicit rounding rule ("choose the nearest cardinal direction")
- **JSON-only response**: `response_format: json_object` prevents prose output

This tight constraint means each frame classification is almost binary — the model has very little room to hallucinate.

---

## Handling Non-Determinism

Temperature=0 is not truly deterministic in practice (floating-point, MoE routing, batching effects cause ~1-3% variance). Mitigations:

1. **Majority-vote smoothing** (window=3): Single misclassified frames are absorbed by neighbors
2. **Algorithmic counting**: The state machine is 100% deterministic — LLM non-determinism only affects the input labels
3. **Contact sheet temporal context**: VLM sees adjacent frames in the same image, giving it implicit sequence context
4. **Confidence filtering**: UNKNOWN cells are interpolated from neighbors rather than propagating errors
5. **Validation checks**: If cumulative angle is inconsistent with count, a warning is logged

---

## Key Challenges & Considerations

### 1. Frame Rate Calibration
- Too low (1 FPS): May miss entire cardinal directions in fast rotations, causing undercounting
- Too high (6+ FPS): Exponential API cost with minimal accuracy gain
- **Recommendation**: 3 FPS is the sweet spot for typical human rotation speeds (2-4s per rotation). Adjust `TARGET_FPS` in `config.py` if rotations are unusually fast.

### 2. Grid Cell Resolution
- 4×4 grids (128px per cell) may be too small for reliable orientation classification
- 2×4 grids (256px per cell) are the default — better VLM accuracy at 2× the API calls
- Test with `GRID_COLS=2, GRID_ROWS=4` vs `GRID_COLS=4, GRID_ROWS=4` if accuracy issues arise

### 3. Clothing/Symmetry Ambiguity
- Symmetrical clothing (plain t-shirt) makes FRONT/BACK ambiguous
- The prompt emphasizes shoulder/hip plane geometry, not visual features like logos
- Risk: If the person wears a symmetric outfit and the video has poor depth cues, FRONT/BACK may be swapped consistently — this still gives the correct count as long as it's consistent

### 4. CW vs CCW Detection
- The algorithm infers direction from the plurality of transitions in the first frames
- Risk: If the first few frames are misclassified, direction could be inverted
- Mitigation: Uses all transitions (not just the first), and flags low direction consistency in warnings

### 5. 45° Diagonal Orientations
- The person spends time at diagonal angles (between FRONT and RIGHT_SIDE, etc.)
- The prompt instructs rounding to nearest cardinal, but VLMs may hesitate
- Mitigation: Majority-vote smoothing absorbs these as noise

### 6. Contact Sheet Ordering
- VLM must read cells in the correct order (1 left-to-right, top-to-bottom)
- Risk: VLM may miscount cells or swap indices in JSON response
- Mitigation: JSON validation checks that expected keys (1 through N) are present; missing keys default to UNKNOWN

### 7. API Rate Limits (5 Consecutive Runs)
- The `run_5x.sh` script adds 2-second pauses between runs
- OpenAI's rate limit for gpt-4o-mini is generous (10,000 RPM on paid tiers) but 5 rapid runs of 12 calls each could spike
- Mitigation: Semaphore limits concurrency to 5 simultaneous calls; exponential backoff on 429s

### 8. Cost Estimate
- At 3 FPS, 30-second video: ~90 frames → 12 contact sheets (2×4)
- GPT-4o-mini vision input ~3,000 tokens/sheet × 12 sheets = 36,000 tokens
- Cost: ~$0.005 per run × 5 runs = **< $0.03 total**
- Gemini Flash fallback: even cheaper (~$0.001 per run)

### 9. Partial Rotations
- The counter uses `int(cumulative_angle / 360)` — partial rotations (e.g., 5.7 rotations → 5) are floored
- The validator flags if cumulative angle is >0.5 rotations from the integer count

### 10. Unknown Video Characteristics
- The pipeline auto-detects source FPS via `CAP_PROP_FPS` and adapts sampling
- If the subject leaves frame mid-rotation, those frames classify as UNKNOWN and are interpolated from neighbors

---

## Failure Modes Encountered in Design

| Failure Mode | Why It Happens | Mitigation |
|---|---|---|
| "Count the rotations" prompt → hallucination | LLM invents plausible-sounding numbers without watching carefully | Never ask LLM to count — only to classify per-frame |
| Per-frame API costs too high | 90 frames × $0.01 = $0.90/run | Contact sheet batching reduces to 12 calls |
| Single frame flip causes count-off-by-one | VLM classifies a BACK frame as FRONT at rotation boundary | Majority-vote smoothing window=3 |
| Direction detection wrong → half count | First frames are noisy or ambiguous | Use plurality of all transitions, not just first |
| Response not valid JSON | VLM wraps JSON in markdown or adds explanation | Regex JSON extraction fallback; `json_object` response format |
| Rate limit on 5 consecutive runs | Too many rapid API calls | 2s pause between runs + exponential backoff |

---

## Configuration Reference

All tunable parameters are in `config.py`:

| Parameter | Default | Effect |
|---|---|---|
| `TARGET_FPS` | 3 | Frames sampled per second |
| `FRAME_WIDTH` | 512 | Frame resize width in pixels |
| `GRID_COLS` | 2 | Contact sheet columns |
| `GRID_ROWS` | 4 | Contact sheet rows |
| `PRIMARY_MODEL` | `gpt-4o-mini` | VLM for classification |
| `VLM_TEMPERATURE` | 0 | Determinism setting |
| `VLM_MAX_CONCURRENT` | 5 | Async concurrency limit |
| `SMOOTHING_WINDOW` | 3 | Majority-vote window size |

---

## Dependencies

- `opencv-python` — frame extraction only (no tracking)
- `Pillow` — contact sheet composition
- `openai` — primary VLM (gpt-4o-mini)
- `google-generativeai` — fallback VLM (gemini-2.0-flash)
- `httpx` — async HTTP client
- `python-dotenv` — environment variable loading
- `pytest` — unit testing
