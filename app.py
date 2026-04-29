from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st


@dataclass
class BusMessage:
    timestamp: str
    topic: str
    producer: str
    consumer: str
    priority: str
    latency_s: float
    payload: dict[str, Any]


class SimulatedMessageBus:
    def __init__(self) -> None:
        self._topics: dict[str, deque[BusMessage]] = defaultdict(lambda: deque(maxlen=200))
        self._history: deque[BusMessage] = deque(maxlen=500)
        self._lock = Lock()

    def publish(self, message: BusMessage) -> None:
        with self._lock:
            self._topics[message.topic].appendleft(message)
            self._history.appendleft(message)

    def consume_latest(self, topic: str) -> BusMessage | None:
        with self._lock:
            if not self._topics[topic]:
                return None
            return self._topics[topic][0]

    def history(self) -> list[BusMessage]:
        with self._lock:
            return list(self._history)

    def clear(self) -> None:
        with self._lock:
            self._topics.clear()
            self._history.clear()


@st.cache_resource
def get_bus() -> SimulatedMessageBus:
    return SimulatedMessageBus()


@st.cache_resource
def get_bg_subtractor() -> cv2.BackgroundSubtractor:
    return cv2.createBackgroundSubtractorMOG2(history=240, varThreshold=28, detectShadows=False)


def timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")


def make_track_mask(width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(
        [
            [int(width * 0.35), height],
            [int(width * 0.47), int(height * 0.40)],
            [int(width * 0.53), int(height * 0.40)],
            [int(width * 0.65), height],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [pts], 255)
    return mask


def detect_obstacles(
    frame_bgr: np.ndarray,
    bg_subtractor: cv2.BackgroundSubtractor,
    min_blob_area: int,
    conf_threshold: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    annotated = frame_bgr.copy()
    h, w, _ = frame_bgr.shape
    track_mask = make_track_mask(w, h)

    fg_mask = bg_subtractor.apply(frame_bgr)
    fg_mask = cv2.GaussianBlur(fg_mask, (7, 7), 0)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    fg_mask = cv2.bitwise_and(fg_mask, track_mask)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    fg_mask = cv2.dilate(fg_mask, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[dict[str, Any]] = []

    cv2.polylines(
        annotated,
        [
            np.array(
                [
                    [int(w * 0.35), h],
                    [int(w * 0.47), int(h * 0.40)],
                    [int(w * 0.53), int(h * 0.40)],
                    [int(w * 0.65), h],
                ],
                dtype=np.int32,
            )
        ],
        isClosed=True,
        color=(255, 200, 0),
        thickness=2,
    )

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_blob_area:
            continue
        x, y, ww, hh = cv2.boundingRect(contour)
        confidence = min(0.99, 0.52 + (area / 12000))
        if confidence < conf_threshold:
            continue

        distance_estimate_m = max(5, int(150 - ((y + hh) / h) * 130))
        det = {
            "label": "Obstacle on rail corridor",
            "confidence": round(confidence, 2),
            "bbox": (x, y, x + ww, y + hh),
            "distance_m": distance_estimate_m,
        }
        detections.append(det)
        cv2.rectangle(annotated, (x, y), (x + ww, y + hh), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            f"{det['label']} {det['confidence']:.2f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (0, 50, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        annotated,
        "OpenCV real-time obstacle pipeline (demo)",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated, detections


def infer_risk(detections: list[dict[str, Any]]) -> tuple[str, int]:
    if not detections:
        return "LOW", 180
    min_distance = min(d["distance_m"] for d in detections)
    if min_distance < 30:
        return "CRITICAL", 18
    if min_distance < 60:
        return "HIGH", 45
    return "MEDIUM", 90


def is_effectively_black(frame: np.ndarray, threshold: float = 6.0) -> bool:
    return float(np.mean(frame)) < threshold


def synthetic_frame(width: int, height: int, tick: int, include_obstacle: bool) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (20, 24, 32)

    # Rails / corridor
    pts = np.array(
        [
            [int(width * 0.30), height],
            [int(width * 0.45), int(height * 0.35)],
            [int(width * 0.55), int(height * 0.35)],
            [int(width * 0.70), height],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [pts], (55, 60, 70))
    cv2.polylines(frame, [pts], True, (80, 160, 220), 2)

    # Front stopped train (blue)
    cv2.rectangle(frame, (int(width * 0.46), int(height * 0.56)), (int(width * 0.58), int(height * 0.70)), (255, 120, 60), -1)
    # Rear approaching train (red)
    cv2.rectangle(frame, (int(width * 0.47), int(height * 0.16)), (int(width * 0.57), int(height * 0.30)), (60, 80, 240), -1)

    if include_obstacle:
        x = int(width * 0.40 + (tick % 90) * 2)
        y = int(height * 0.52)
        cv2.rectangle(frame, (x, y), (x + 65, y + 35), (60, 200, 60), -1)
        cv2.putText(frame, "Taxi", (x + 8, y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 30, 20), 1, cv2.LINE_AA)

    cv2.putText(frame, "Synthetic fallback stream", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (220, 220, 220), 2, cv2.LINE_AA)
    return frame


def publish_pipeline_messages(
    bus: SimulatedMessageBus,
    track_id: str,
    detections: list[dict[str, Any]],
    risk: str,
    eta_s: int,
    baseline_latency: float,
    ai_latency: float,
) -> None:
    if not detections:
        return

    now = timestamp()
    vision_msg = BusMessage(
        timestamp=now,
        topic="vision.alert",
        producer="vision-node",
        consumer="control-center",
        priority=risk,
        latency_s=0.45,
        payload={
            "track_id": track_id,
            "risk": risk,
            "eta_to_impact_s": eta_s,
            "detection_count": len(detections),
            "nearest_distance_m": min(d["distance_m"] for d in detections),
        },
    )
    bus.publish(vision_msg)

    control_cmd = BusMessage(
        timestamp=now,
        topic="control.command",
        producer="control-center",
        consumer="operator-cab",
        priority=risk,
        latency_s=ai_latency,
        payload={
            "track_id": track_id,
            "action": "HOLD + EMERGENCY_BRAKE" if risk in {"HIGH", "CRITICAL"} else "REDUCE_SPEED",
            "eta_to_impact_s": eta_s,
            "legacy_latency_s": baseline_latency,
            "ai_latency_s": ai_latency,
            "time_saved_s": max(0.0, baseline_latency - ai_latency),
        },
    )
    bus.publish(control_cmd)

    ack = BusMessage(
        timestamp=now,
        topic="operator.ack",
        producer="operator-cab",
        consumer="control-center",
        priority=risk,
        latency_s=1.2,
        payload={
            "track_id": track_id,
            "status": "ACKNOWLEDGED",
            "brake_command_executed": risk in {"HIGH", "CRITICAL"},
        },
    )
    bus.publish(ack)


def init_state() -> None:
    if "run_stats" not in st.session_state:
        st.session_state.run_stats = []
    if "runner_active" not in st.session_state:
        st.session_state.runner_active = False
    if "cap" not in st.session_state:
        st.session_state.cap = None
    if "temp_video_path" not in st.session_state:
        st.session_state.temp_video_path = None
    if "use_synthetic" not in st.session_state:
        st.session_state.use_synthetic = False
    if "synthetic_tick" not in st.session_state:
        st.session_state.synthetic_tick = 0
    if "black_frames" not in st.session_state:
        st.session_state.black_frames = 0
    if "run_started_at" not in st.session_state:
        st.session_state.run_started_at = None
    if "processed_frames" not in st.session_state:
        st.session_state.processed_frames = 0
    if "critical_events" not in st.session_state:
        st.session_state.critical_events = 0
    if "source_status" not in st.session_state:
        st.session_state.source_status = "Idle"
    if "last_risk" not in st.session_state:
        st.session_state.last_risk = "LOW"
    if "last_detections" not in st.session_state:
        st.session_state.last_detections = []


def cleanup_video_state() -> None:
    cap = st.session_state.cap
    if cap is not None:
        cap.release()
    st.session_state.cap = None
    path = st.session_state.temp_video_path
    if path:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass
    st.session_state.temp_video_path = None


def stop_runner(reason: str | None = None) -> None:
    if reason:
        st.session_state.source_status = reason
    st.session_state.runner_active = False
    cleanup_video_state()
    st.session_state.run_started_at = None
    st.session_state.processed_frames = 0
    st.session_state.critical_events = 0
    st.session_state.black_frames = 0


def start_runner(source: str, uploaded: Any, synthetic_obstacle: bool) -> None:
    cleanup_video_state()
    st.session_state.runner_active = True
    st.session_state.synthetic_tick = 0
    st.session_state.black_frames = 0
    st.session_state.processed_frames = 0
    st.session_state.critical_events = 0
    st.session_state.run_started_at = time.time()
    st.session_state.use_synthetic = False
    st.session_state.synthetic_obstacle = synthetic_obstacle

    if source == "Webcam (index 0)":
        cap = cv2.VideoCapture(0)
        if cap is None or not cap.isOpened():
            st.session_state.use_synthetic = True
            st.session_state.source_status = "Webcam unavailable. Running synthetic fallback stream."
            st.session_state.cap = None
            return
        st.session_state.cap = cap
        st.session_state.source_status = "Active source: Webcam (index 0)"
        return

    if uploaded is None:
        st.session_state.runner_active = False
        st.session_state.source_status = "Upload a video file before starting."
        return

    temp_video_path = f"/tmp/rail_demo_{int(time.time())}.mp4"
    with open(temp_video_path, "wb") as f:
        f.write(uploaded.read())
    cap = cv2.VideoCapture(temp_video_path)
    if cap is None or not cap.isOpened():
        st.session_state.runner_active = False
        st.session_state.source_status = "Unable to open uploaded video source."
        st.session_state.temp_video_path = temp_video_path
        cleanup_video_state()
        return
    st.session_state.cap = cap
    st.session_state.temp_video_path = temp_video_path
    st.session_state.source_status = "Active source: Uploaded video file"


def main() -> None:
    st.set_page_config(page_title="AI Rail Signalling - OpenCV + Message Bus", layout="wide")
    init_state()
    bus = get_bus()
    bg_subtractor = get_bg_subtractor()

    st.title("AI-Assisted Rail Signalling: OpenCV + Simulated Message Bus")
    st.caption(
        "This demo processes live frames with OpenCV and publishes simulated control-center messages in a pub/sub flow."
    )
    view_mode = st.radio("Display", ["Live Camera", "Analytics"], horizontal=True)

    with st.sidebar:
        st.header("Pipeline Controls")
        source = st.selectbox("Video source", ["Webcam (index 0)", "Uploaded video file"])
        uploaded = st.file_uploader("Upload MP4/MOV/AVI", type=["mp4", "mov", "avi"])
        synthetic_obstacle = st.toggle("Synthetic fallback includes obstacle", value=True)
        track_id = st.selectbox("Track segment", ["BKS-TIMUR-A", "BKS-TIMUR-B", "BKS-TIMUR-C"])
        min_blob_area = st.slider("Minimum obstacle blob area", 200, 6000, 1200, 100)
        conf_threshold = st.slider("Detection confidence threshold", 0.50, 0.95, 0.70, 0.01)
        baseline_latency = st.slider("Legacy signalling latency (s)", 30, 240, 120, 10)
        ai_latency = st.slider("AI-assisted command latency (s)", 2, 45, 8, 1)
        duration_s = st.slider("Run duration per session (s)", 5, 120, 30, 1)
        max_fps = st.slider("Processing FPS cap", 5, 30, 12, 1)
        start = st.button("Start real-time run", width="stretch")
        stop = st.button("Stop real-time run", width="stretch", type="secondary")
        clear_bus = st.button("Clear message bus history", width="stretch")
        if start:
            start_runner(source=source, uploaded=uploaded, synthetic_obstacle=synthetic_obstacle)
        if stop:
            stop_runner("Run stopped by user.")
        if clear_bus:
            bus.clear()
            st.session_state.run_stats = []

    should_rerun = False
    st.info(st.session_state.source_status)
    current_frame_rgb: np.ndarray | None = None
    if st.session_state.runner_active:
        frame_interval = 1.0 / float(max_fps)
        frame_start = time.time()
        elapsed = frame_start - float(st.session_state.run_started_at or frame_start)

        if elapsed >= duration_s:
            fps = st.session_state.processed_frames / elapsed if elapsed > 0 else 0.0
            st.session_state.run_stats.insert(
                0,
                {
                    "timestamp": timestamp(),
                    "frames": st.session_state.processed_frames,
                    "elapsed_s": round(elapsed, 2),
                    "avg_fps": round(fps, 2),
                    "critical_events": st.session_state.critical_events,
                    "latency_saved_s": max(0.0, baseline_latency - ai_latency),
                },
            )
            st.session_state.run_stats = st.session_state.run_stats[:40]
            stop_runner("Run completed.")
        else:
            if st.session_state.use_synthetic:
                frame = synthetic_frame(
                    960,
                    540,
                    st.session_state.synthetic_tick,
                    include_obstacle=bool(st.session_state.get("synthetic_obstacle", True)),
                )
                st.session_state.synthetic_tick += 1
            else:
                cap = st.session_state.cap
                ok, frame = (cap.read() if cap is not None else (False, None))
                if not ok or frame is None:
                    if source == "Webcam (index 0)":
                        st.session_state.use_synthetic = True
                        st.session_state.source_status = "Webcam feed interrupted. Switched to synthetic fallback stream."
                        should_rerun = True
                        frame = synthetic_frame(
                            960,
                            540,
                            st.session_state.synthetic_tick,
                            include_obstacle=bool(st.session_state.get("synthetic_obstacle", True)),
                        )
                        st.session_state.synthetic_tick += 1
                    stop_runner("Uploaded video finished or became unavailable.")
                    should_rerun = True
                    frame = None
                    detections: list[dict[str, Any]] = []
                else:
                    if source == "Webcam (index 0)" and is_effectively_black(frame):
                        st.session_state.black_frames += 1
                        if st.session_state.black_frames >= 8:
                            st.session_state.use_synthetic = True
                            st.session_state.source_status = (
                                "Webcam frames are black (likely permission/device issue). "
                                "Switched to synthetic fallback stream."
                            )
                            should_rerun = True
                            frame = synthetic_frame(
                                960,
                                540,
                                st.session_state.synthetic_tick,
                                include_obstacle=bool(st.session_state.get("synthetic_obstacle", True)),
                            )
                            st.session_state.synthetic_tick += 1
                    else:
                        st.session_state.black_frames = 0
            if frame is not None:
                annotated, detections = detect_obstacles(
                    frame_bgr=frame,
                    bg_subtractor=bg_subtractor,
                    min_blob_area=min_blob_area,
                    conf_threshold=conf_threshold,
                )
                risk, eta_s = infer_risk(detections)
                st.session_state.last_risk = risk
                st.session_state.last_detections = detections
                if risk == "CRITICAL":
                    st.session_state.critical_events += 1

                publish_pipeline_messages(
                    bus=bus,
                    track_id=track_id,
                    detections=detections,
                    risk=risk,
                    eta_s=eta_s,
                    baseline_latency=baseline_latency,
                    ai_latency=ai_latency,
                )

                current_frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.session_state.processed_frames += 1

            remaining = frame_interval - (time.time() - frame_start)
            if remaining > 0:
                time.sleep(remaining)
            should_rerun = True
    if view_mode == "Live Camera":
        st.subheader("Live Obstacle Feed")
        if current_frame_rgb is not None:
            st.image(current_frame_rgb, channels="RGB", width="stretch")
        elif st.session_state.last_detections:
            st.caption("Live frame is idle. Showing latest detection state.")
        else:
            st.caption("Press Start to begin streaming frames.")

        st.metric("Current risk", st.session_state.last_risk)
        st.dataframe(pd.DataFrame(st.session_state.last_detections), width="stretch", hide_index=True)
    else:
        st.subheader("Analytics")
        left, right = st.columns([2, 3], gap="large")

        with left:
            st.markdown("**Run Statistics**")
            st.dataframe(pd.DataFrame(st.session_state.run_stats), width="stretch", hide_index=True)

        with right:
            st.markdown("**Simulated Message Bus Timeline**")
            history = [asdict(m) for m in bus.history()]
            hist_df = pd.DataFrame(history)
            st.dataframe(hist_df, width="stretch", hide_index=True)
            if not hist_df.empty:
                delay_df = hist_df[["timestamp", "latency_s"]].copy().rename(columns={"timestamp": "tick"})
                st.line_chart(delay_df.set_index("tick"), width="stretch")

    latest_cmd = bus.consume_latest("control.command")
    if latest_cmd is not None:
        st.subheader("Latest Control-Center Command")
        st.code(
            (
                f"[{latest_cmd.timestamp}] CONTROL->{latest_cmd.consumer}\n"
                f"Track: {latest_cmd.payload['track_id']}\n"
                f"Action: {latest_cmd.payload['action']}\n"
                f"Risk: {latest_cmd.priority}\n"
                f"Legacy latency: {latest_cmd.payload['legacy_latency_s']}s\n"
                f"AI latency: {latest_cmd.payload['ai_latency_s']}s\n"
                f"Time saved: {latest_cmd.payload['time_saved_s']:.1f}s"
            ),
            language="text",
        )

    if should_rerun and st.session_state.runner_active:
        st.rerun()


if __name__ == "__main__":
    main()
