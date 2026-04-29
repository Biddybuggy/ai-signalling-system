# User Guide

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open `http://localhost:8501`.

## Basic workflow

1. Choose a video source:
   - `Webcam (index 0)`, or
   - `Uploaded video file`.
2. Set detection and latency parameters.
3. Click `Start real-time run`.
4. Watch:
   - annotated frame stream,
   - current risk metric,
   - detection table,
   - message bus timeline,
   - latest control-center command.
5. Click `Stop real-time run` any time.

## Controls explained

- `Synthetic fallback includes obstacle`:
  - Affects synthetic stream content when fallback is active.
- `Minimum obstacle blob area`:
  - Ignores small moving artifacts/noise below this contour size.
- `Detection confidence threshold`:
  - Filters low-confidence detections.
- `Legacy signalling latency (s)`:
  - Baseline delay for traditional/manual signalling path.
- `AI-assisted command latency (s)`:
  - Faster response path used in mock command generation.
- `Run duration per session (s)`:
  - Auto-stop timer for a running session.
- `Processing FPS cap`:
  - Upper bound on processed frames per second.

## Reading the output

- `Current risk`:
  - Based on nearest estimated obstacle distance.
- `Run Statistics`:
  - Frames processed, elapsed time, average FPS, critical event count, modeled time saved.
- `Simulated Message Bus Timeline`:
  - Event trace for `vision.alert`, `control.command`, and `operator.ack`.
- `Latest Control-Center Command`:
  - Most recent operator action payload (track, action, latency deltas).

## Troubleshooting

- Blank or black webcam frames:
  - App auto-switches to synthetic fallback stream.
- Uploaded video fails:
  - Re-export video as MP4 (H.264) and re-upload.
- Slow frame rate:
  - Lower `Processing FPS cap`, reduce resolution, or use shorter clips.

## Demo script for presentations

1. Start run with webcam source.
2. Show automatic fallback warning if webcam is blocked.
3. Highlight real-time detection overlays.
4. Compare legacy vs AI latency in command payload.
5. Point to message bus timeline as control-center audit trail.
