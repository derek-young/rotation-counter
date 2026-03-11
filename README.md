# LLM-Based Rotation Counter

This system counts the number of full 360° human body rotations in a video using strictly LLMs/VLMs, bypassing traditional computer vision tracking dependencies.

**Example Usage**

Single run:
```bash
% python3 main.py rotationsTest.qt
# → 5
```

Five run validation:
```bash
% ./scripts/run_5x.sh rotationsTest.qt
```

---

## High-Level Approach

1. Temporal Sampling: Extract video frames at 3 FPS (configurable)
2. Context Batching: Compose contact sheet grids (4 x 2)
3. State Classification: Classify each contact sheet quadrant as FRONT | RIGHT_SIDE | BACK | LEFT_SIDE
4. Heuristic Logic: Count rotations based on state transitions: Front -> Back -> Front = 1 rotation


### Video Frame Extraction

For the provided sample video, the subject rotates relatively slowly, so to optimize inference costs, we sample at 3 frames per second. This is configurable; in a production environment, an adaptive sampling value depending on motion velocity would be ideal.

Cost-Accuracy Tradeoff: The state machine defines a rotation as F -> B -> F. If the FPS is too low and a key orientation is missed, the system may under-count.


### Contact Sheet Batching

To mitigate the latency of round-trip network requests, the frames are resized to 512px and tiled into a N x M grid. This reduces the number of network requests by a factor of N x M. Using the default values, this approach provided a ~7× improvement in both throughput and API economy.


### Classification

The VLM receives a grid of numbered frames and classifies each frame's orientation using a strictly constrained system prompt:

- Enumerated Output: Only 5 valid labels (FRONT, RIGHT_SIDE, BACK, LEFT_SIDE, UNKNOWN).
- Angular Quantization: Explicit rounding rule to "choose the nearest cardinal direction."
- Structured Output: json_object response format prevents prose hallucination.

By narrowing the label space, the model functions as a reliable discrete classifier. To improve accuracy, the prompt guides the LLM through an "Anatomical Audit": reasoning about the head, torso, arms, and hips individually before committing to a final label. This Chain-of-Thought approach yielded a 10% accuracy improvement over single-shot classification.


### Handling Non-Determinism

The LLM is prompted with a temperature of 0 to minimize stochastic variance. Additionally, a fixed seed is passed to the API to ensure reproducible token sampling.

In testing, the F -> B -> F heuristic remained accurate across all iterations. For high-stakes production use, a "Majority Vote" ensemble (running 3 parallel classifications) would further harden the pipeline.


### Temporal Tracking Failures & Iteration

The first approach attempted to calculate cumulative angular displacement. However, while the VLM excelled at Front/Back detection, it struggled to reliably distinguish Left vs. Right profiles. This caused the cumulative angle approach to fail due to "direction-flipping" hallucinations.

By pivoting to the more robust F -> B -> F state-transition approach, inference costs were further reduced because the image detail parameter could be dropped to "low." This reduced the inference latency by half without compromising the detection of these primary cardinal orientations.

---
## Key Performance Metrics
*Metrics based on `rotationsTest.qt` (~10s) using the `gpt-5.4-2026-03-05` model with "Low Detail" vision settings. Averaged over 10 runs.*


| Metric | Average Value | Description |
| :--- | :--- | :--- |
| **Total Execution Time** | ~2.5 seconds | End-to-end processing (Extraction + Batching + Inference) |
| **Total Tokens** | ~8,400 tokens | Total tokens per analysis (input + output) |
| **Classification Accuracy** | ~55% | Frame-level orientation classification accuracy |
| **Count Accuracy** | 100% | Correct rotation count (5/5) across all 10 runs |
| **Cost per Analysis** | ~$0.024 USD | ~$0.02 input + ~$0.004 output (at $2.50/1M input, $15.00/1M output) |

---

## Try It Yourself

```bash
# Clone the repo
git clone https://github.com/derek-young/rotation-counter.git

# Spin up virtual env
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Run
python3 main.py rotationsTest.qt

# Run 5 consecutive times (consistency test)
chmod +x scripts/run_5x.sh
./scripts/run_5x.sh rotationsTest.qt

# Unit tests
python -m pytest tests/ -v
```