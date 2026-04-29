# AI-Assisted Train Signalling Mock-up

This project is a **non-production safety demo** that now includes:

- A **real-time OpenCV video pipeline** for obstacle detection inside a virtual rail corridor.
- A **simulated pub/sub message bus** to model communication from vision nodes to control center and train operators.
- A latency comparison between **legacy signalling delay** and **AI-assisted dispatch delay**.
- Responsive run controls with **start/stop real-time execution**.

It is intentionally simplified and exists for concept visualization only.

## License

This project is licensed under the MIT License. See `LICENSE`.

## Architecture diagram

![Runtime architecture](docs/architecture-diagram.png)

## System architecture

The app (`app.py`) runs as a Streamlit dashboard with these logical components:

1. **Video Source Layer**
   - Webcam (`cv2.VideoCapture(0)`) or uploaded video file.
   - Frames are processed at a configurable FPS cap.

2. **Vision Pipeline (OpenCV)**
   - Background subtraction using `cv2.createBackgroundSubtractorMOG2`.
   - Foreground cleanup with blur, thresholding, morphology, and dilation.
   - Region-of-interest (ROI) mask for a virtual rail corridor polygon.
   - Contour extraction to detect moving/foreground blobs.
   - Bounding boxes + confidence estimation + approximate distance estimate.

3. **Risk Inference**
   - Uses nearest estimated obstacle distance:
     - `CRITICAL`: obstacle very near
     - `HIGH`/`MEDIUM`/`LOW`: progressively safer states
   - Produces a rough ETA-to-impact for communication decisions.

4. **Simulated Message Bus**
   - In-memory pub/sub with topics and bounded queues.
   - Thread-safe via `threading.Lock`.
   - Message metadata includes producer, consumer, priority, payload, and simulated latency.

5. **Control-Center Communication Model**
   - `vision.alert`: sensor/vision event to control center.
   - `control.command`: control center action to operator cab.
   - `operator.ack`: acknowledgement back to control center.
   - Dashboard renders message timeline and latest command.

## Message bus technical model

The bus is implemented by `SimulatedMessageBus`:

- `publish(message)`: pushes a `BusMessage` into topic queue and global history.
- `consume_latest(topic)`: reads latest topic message.
- `history()`: returns a chronological snapshot for UI timeline.
- `clear()`: resets all topics and history.

`BusMessage` fields:

- `timestamp`: local wall-clock time for demo traceability.
- `topic`: logical channel name (`vision.alert`, `control.command`, `operator.ack`).
- `producer` / `consumer`: sender and intended recipient.
- `priority`: mapped from risk level.
- `latency_s`: simulated message transit/processing delay.
- `payload`: domain details (track segment, ETA, action, time saved).

## Detection pipeline technicalities

Each frame follows this sequence:

1. Build a **track ROI mask** (polygon in image space).
2. Run MOG2 to separate dynamic foreground from background.
3. Smooth mask (`GaussianBlur`) and binarize (`threshold`).
4. Keep only pixels inside rail ROI (`bitwise_and`).
5. Remove noise (`MORPH_OPEN`) and merge object pixels (`dilate`).
6. Find contours and filter by minimum blob area.
7. Convert contour area into a confidence heuristic.
8. Estimate distance by vertical position in frame (closer objects appear lower).
9. Draw boxes and labels on the frame for operator visibility.

This is a practical computer-vision baseline, not a certified obstacle classifier.

## Making alerts less error-prone (dwell + planned-stop context)

Instead of treating “any blob exists” as an obstacle, the demo upgrades the decision layer with two extra realism features:

- **Multi-frame tracking and dwell time**: detected blobs are associated over time and only escalated when a track stays (nearly) stationary for a configurable duration.
- **Blocking evidence**: a track must also occupy enough of a “crossing ROI” region before it can become an obstacle.

To model the “train told to stop briefly” case, the demo also includes a **planned stop suppression** mechanism:
- If a “train-like” tracked object stays within an expected planned-stop window, it is *suppressed* (not treated as an obstacle).
- If it exceeds the expected window (simulating a stuck train), it triggers the higher-risk dispatch logic.

## Latency model

The app tracks two configurable delays:

- **Legacy signalling latency**: slower human/manual or legacy dispatch path.
- **AI-assisted latency**: faster machine-assisted alert/dispatch path.

For each control command, payload includes:

- `legacy_latency_s`
- `ai_latency_s`
- `time_saved_s = max(legacy - ai, 0)`

This allows the control-center panel to show potential response-time gains.

## Running the project

From this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL shown by Streamlit (typically `http://localhost:8501`).

## Project documentation

- Architecture and design: `docs/ARCHITECTURE.md`
- Hands-on usage and troubleshooting: `docs/USER_GUIDE.md`
- Main implementation: `app.py`

## Dashboard controls

- **Video source**: webcam or uploaded MP4/MOV/AVI.
- **Minimum obstacle blob area**: contour size threshold to suppress noise.
- **Detection confidence threshold**: filter weak detections.
- **Legacy / AI latency sliders**: communication-delay scenario modeling.
- **Run duration + FPS cap**: bounded real-time session control.
- **Start real-time run**: execute live frame loop and publish bus messages.
- **Stop real-time run**: interrupt streaming immediately.
- **Clear message bus history**: reset timeline and run statistics.

## Limitations and safety disclaimer

This code is not railway-safe and should not be connected to real signalling hardware.
Main limitations include:

- No sensor redundancy or formal fail-safe design.
- No verified braking dynamics or train-interlocking logic.
- No authenticated networking, cybersecurity, or fault tolerance.
- No standards compliance (for example EN 50126/50128/50129 processes).
- Vision confidence and distance are heuristic approximations.

## Suggested next upgrades

1. Replace heuristic detection with trained detection models (for example YOLO/RT-DETR).
2. Add sensor fusion simulation (camera + lidar/radar + track circuits).
3. Move bus to a real broker (MQTT, NATS, Kafka) with schema validation.
4. Introduce deterministic state machines for interlocking and braking.
5. Add replayable incident scenarios and evaluation metrics (recall, false alarms, mean alert lead time).