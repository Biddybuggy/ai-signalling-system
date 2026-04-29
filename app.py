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


@dataclass
class TrackState:
    track_id: int
    centroid_x: float
    centroid_y: float
    last_seen_ts: float
    dwell_s: float
    last_speed_px_s: float
    occupancy_ratio: float
    distance_m: int
    area_pixels: float
    last_bbox: tuple[int, int, int, int]


def compute_crossing_roi(width: int, height: int) -> tuple[int, int, int, int]:
    """
    Approximate "level crossing / blocking area" inside the camera frame.
    This is used purely for demo logic and synthetic visuals.
    """
    x1 = int(width * 0.33)
    y1 = int(height * 0.42)
    x2 = int(width * 0.67)
    y2 = int(height * 0.72)
    return x1, y1, x2, y2


def bbox_area(bbox: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def centroid_of_bbox(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def intersection_area(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return float((ix2 - ix1) * (iy2 - iy1))


def detect_obstacles(
    frame_bgr: np.ndarray,
    bg_subtractor: cv2.BackgroundSubtractor,
    min_blob_area: int,
    conf_threshold: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    annotated = frame_bgr.copy()
    h, w, _ = frame_bgr.shape
    track_mask = make_track_mask(w, h)
    crossing_roi = compute_crossing_roi(w, h)

    fg_mask = bg_subtractor.apply(frame_bgr)
    fg_mask = cv2.GaussianBlur(fg_mask, (7, 7), 0)
    # Lower threshold to keep "present but not moving" blobs visible longer.
    _, fg_mask = cv2.threshold(fg_mask, 160, 255, cv2.THRESH_BINARY)
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
        det_bbox = (x, y, x + ww, y + hh)
        det_area = bbox_area(det_bbox)
        occ_ratio = 0.0
        if det_area > 1e-6:
            occ_ratio = intersection_area(det_bbox, crossing_roi) / det_area
        det = {
            "label": "Foreground blob (vision demo)",
            "confidence": round(confidence, 2),
            "bbox": det_bbox,
            "centroid": centroid_of_bbox(det_bbox),
            "area_pixels": float(det_area),
            "distance_m": distance_estimate_m,
            "occupancy_ratio": float(round(occ_ratio, 3)),
        }
        detections.append(det)
        cv2.rectangle(annotated, (x, y), (x + ww, y + hh), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            f"blob {det['confidence']:.2f}",
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
    # Draw crossing ROI for operator explainability.
    x1, y1, x2, y2 = crossing_roi
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 255), 2)
    cv2.putText(
        annotated,
        "Crossing ROI",
        (x1 + 4, max(20, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 200, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated, detections


def update_tracks(
    detections: list[dict[str, Any]],
    tracks: dict[int, TrackState],
    next_track_id: int,
    now_ts: float,
    dt_s: float,
    match_distance_px: float,
    stationary_speed_px_s: float,
) -> tuple[dict[int, TrackState], int]:
    """
    Greedy nearest-centroid association.
    The goal is not perfect tracking, but stable dwell-time accumulation.
    """
    det_centroids = [
        (i, float(d["centroid"][0]), float(d["centroid"][1])) for i, d in enumerate(detections)
    ]
    assigned_track_ids: set[int] = set()
    assigned_det_ids: set[int] = set()

    for det_id, cx, cy in det_centroids:
        best_track_id: int | None = None
        best_dist = float("inf")
        for track_id, trk in tracks.items():
            if track_id in assigned_track_ids:
                continue
            dist = ((trk.centroid_x - cx) ** 2 + (trk.centroid_y - cy) ** 2) ** 0.5
            if dist < best_dist and dist <= match_distance_px:
                best_dist = dist
                best_track_id = track_id

        if best_track_id is None:
            continue

        trk = tracks[best_track_id]
        # Update speed estimate based on pixel movement.
        speed_px_s = 0.0
        if dt_s > 1e-6:
            speed_px_s = best_dist / dt_s

        stationary = speed_px_s <= stationary_speed_px_s
        new_dwell = trk.dwell_s + dt_s if stationary else 0.0

        det_bbox = detections[det_id]["bbox"]
        tracks[best_track_id] = TrackState(
            track_id=best_track_id,
            centroid_x=cx,
            centroid_y=cy,
            last_seen_ts=now_ts,
            dwell_s=new_dwell,
            last_speed_px_s=float(speed_px_s),
            occupancy_ratio=float(detections[det_id].get("occupancy_ratio", 0.0)),
            distance_m=int(detections[det_id].get("distance_m", 180)),
            area_pixels=float(detections[det_id].get("area_pixels", 0.0)),
            last_bbox=tuple(det_bbox),
        )
        assigned_track_ids.add(best_track_id)
        assigned_det_ids.add(det_id)

    # Create new tracks for unassigned detections.
    for det_id, cx, cy in det_centroids:
        if det_id in assigned_det_ids:
            continue
        det = detections[det_id]
        det_bbox = det["bbox"]
        new_track = TrackState(
            track_id=next_track_id,
            centroid_x=cx,
            centroid_y=cy,
            last_seen_ts=now_ts,
            dwell_s=0.0,
            last_speed_px_s=0.0,
            occupancy_ratio=float(det.get("occupancy_ratio", 0.0)),
            distance_m=int(det.get("distance_m", 180)),
            area_pixels=float(det.get("area_pixels", 0.0)),
            last_bbox=tuple(det_bbox),
        )
        tracks[next_track_id] = new_track
        next_track_id += 1

    # Decay tracks that haven't been seen recently.
    # This prevents stale tracks from accumulating dwell indefinitely.
    max_track_age_s = 5.0
    to_delete = [tid for tid, trk in tracks.items() if now_ts - trk.last_seen_ts > max_track_age_s]
    for tid in to_delete:
        tracks.pop(tid, None)

    return tracks, next_track_id


def obstacle_tracks_from_state(
    tracks: dict[int, TrackState],
    crossing_occupancy_threshold: float,
    stuck_dwell_threshold_s: float,
    planned_stop: dict[str, Any] | None,
    train_area_threshold: float,
    planned_stop_tolerance_s: float,
) -> tuple[list[TrackState], list[dict[str, Any]]]:
    obstacle_tracks: list[TrackState] = []
    obstacle_detections_for_ui: list[dict[str, Any]] = []

    for trk in tracks.values():
        is_stationary_long = trk.dwell_s >= stuck_dwell_threshold_s
        is_blocking = trk.occupancy_ratio >= crossing_occupancy_threshold
        if not (is_stationary_long and is_blocking):
            continue

        # "Planned stop" suppression: if this looks like a train and the stop
        # is within an expected window, don't treat it as an obstacle.
        if planned_stop is not None:
            is_train_like = trk.area_pixels >= train_area_threshold
            if is_train_like:
                expected = float(planned_stop.get("expected_stop_s", 0.0))
                if trk.dwell_s <= expected + float(planned_stop_tolerance_s):
                    continue

        obstacle_tracks.append(trk)
        x1, y1, x2, y2 = trk.last_bbox
        obstacle_detections_for_ui.append(
            {
                "track_id": trk.track_id,
                "label": "Obstacle (tracked, persisted)",
                "confidence": round(min(0.99, 0.5 + trk.occupancy_ratio * 0.5), 2),
                "bbox": (x1, y1, x2, y2),
                "distance_m": trk.distance_m,
                "occupancy_ratio": round(trk.occupancy_ratio, 3),
                "dwell_s": round(trk.dwell_s, 2),
                "speed_px_s": round(trk.last_speed_px_s, 2),
                "area_pixels": int(trk.area_pixels),
            }
        )

    return obstacle_tracks, obstacle_detections_for_ui


def infer_risk_from_obstacles(
    obstacle_tracks: list[TrackState],
) -> tuple[str, int]:
    if not obstacle_tracks:
        return "LOW", 180
    nearest_distance = min(t.distance_m for t in obstacle_tracks)
    # Use dwell+occupancy outcome for severity, and distance for ETA roughness.
    max_dwell = max(t.dwell_s for t in obstacle_tracks)
    if max_dwell >= 90 or nearest_distance < 30:
        return "CRITICAL", 18
    if max_dwell >= 40 or nearest_distance < 60:
        return "HIGH", 45
    return "MEDIUM", 90


def is_effectively_black(frame: np.ndarray, threshold: float = 6.0) -> bool:
    return float(np.mean(frame)) < threshold


def synthetic_frame(
    width: int,
    height: int,
    tick: int,
    elapsed_s: float,
    include_taxi: bool,
    taxi_behavior: str,
    train_actual_stop_s: float,
    train_moving_speed_px_s: float,
) -> np.ndarray:
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

    # Train (blue). It stays "stopped" until `elapsed_s >= train_actual_stop_s`,
    # then moves upward to create motion evidence (so dwell time doesn't keep growing).
    train_y0_stop = int(height * 0.56)
    train_y1_stop = int(height * 0.70)
    if elapsed_s < train_actual_stop_s:
        train_y0 = train_y0_stop
        train_y1 = train_y1_stop
    else:
        dt = float(elapsed_s - train_actual_stop_s)
        offset = int(dt * train_moving_speed_px_s)
        train_y0 = max(0, train_y0_stop - offset)
        train_y1 = max(0, train_y1_stop - offset)
    cv2.rectangle(
        frame,
        (int(width * 0.46), train_y0),
        (int(width * 0.58), train_y1),
        (255, 120, 60),
        -1,
    )

    # Rear approaching train (red) - kept mostly static for demo simplicity.
    cv2.rectangle(
        frame,
        (int(width * 0.47), int(height * 0.16)),
        (int(width * 0.57), int(height * 0.30)),
        (60, 80, 240),
        -1,
    )

    # Taxi in the crossing region. It can either "pass quickly" (moving)
    # or be "stuck" (stationary) to trigger dwell-based obstruction logic.
    if include_taxi:
        y = int(height * 0.52)
        if taxi_behavior == "stuck":
            x = int(width * 0.40)
        else:
            # Passing behavior: sweep across the ROI at a reasonable speed.
            # (tick is used to create motion consistency even when FPS varies.)
            x = int(width * 0.25 + (tick % 120) * 3)
        cv2.rectangle(frame, (x, y), (x + 65, y + 35), (60, 200, 60), -1)
        cv2.putText(
            frame,
            "Taxi",
            (x + 8, y + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (20, 30, 20),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        frame,
        "Synthetic fallback stream",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return frame


def publish_vision_alert(
    bus: SimulatedMessageBus,
    track_id: str,
    risk: str,
    eta_s: int,
    detections: list[dict[str, Any]],
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

def publish_control_dispatch(
    bus: SimulatedMessageBus,
    track_id: str,
    risk: str,
    eta_s: int,
    baseline_latency: float,
    ai_latency: float,
) -> None:
    # Only send dispatch commands for higher risk to reduce false alarms.
    if risk not in {"HIGH", "CRITICAL"}:
        return

    now = timestamp()
    control_cmd = BusMessage(
        timestamp=now,
        topic="control.command",
        producer="control-center",
        consumer="operator-cab",
        priority=risk,
        latency_s=ai_latency,
        payload={
            "track_id": track_id,
            "action": "HOLD + EMERGENCY_BRAKE",
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
            "brake_command_executed": True,
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
    if "tracks" not in st.session_state:
        st.session_state.tracks = {}
    if "next_track_id" not in st.session_state:
        st.session_state.next_track_id = 1
    if "last_frame_ts" not in st.session_state:
        st.session_state.last_frame_ts = None
    if "planned_stop" not in st.session_state:
        st.session_state.planned_stop = None
    if "vision_params" not in st.session_state:
        st.session_state.vision_params = {}
    if "last_obstacle_detections" not in st.session_state:
        st.session_state.last_obstacle_detections = []
    if "last_detection_summaries" not in st.session_state:
        st.session_state.last_detection_summaries = []
    if "synthetic_taxi_behavior" not in st.session_state:
        st.session_state.synthetic_taxi_behavior = "passing"
    if "synthetic_include_taxi" not in st.session_state:
        st.session_state.synthetic_include_taxi = True
    if "synthetic_train_actual_stop_s" not in st.session_state:
        st.session_state.synthetic_train_actual_stop_s = 9999.0
    if "synthetic_train_moving_speed_px_s" not in st.session_state:
        st.session_state.synthetic_train_moving_speed_px_s = 120.0


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
    st.session_state.tracks = {}
    st.session_state.next_track_id = 1
    st.session_state.last_frame_ts = None
    st.session_state.last_obstacle_detections = []


def start_runner(
    source: str,
    uploaded: Any,
    include_taxi: bool,
    taxi_behavior: str,
    planned_stop_mode: str,
    expected_stop_s: float,
    actual_stop_s: float,
    stuck_dwell_threshold_s: float,
    crossing_occupancy_threshold: float,
    planned_stop_tolerance_s: float,
    train_area_threshold: float,
    stationary_speed_px_s: float,
    match_distance_px: float,
) -> None:
    cleanup_video_state()
    st.session_state.runner_active = True
    st.session_state.synthetic_tick = 0
    st.session_state.black_frames = 0
    st.session_state.processed_frames = 0
    st.session_state.critical_events = 0
    st.session_state.run_started_at = time.time()
    st.session_state.use_synthetic = False

    # Reset tracking.
    st.session_state.tracks = {}
    st.session_state.next_track_id = 1
    st.session_state.last_frame_ts = None
    st.session_state.last_obstacle_detections = []

    # Set planned-stop suppression context (train stops are "intentional" within expected duration).
    if planned_stop_mode == "None":
        st.session_state.planned_stop = None
        st.session_state.synthetic_train_actual_stop_s = 0.0
    elif planned_stop_mode == "Planned stop within expected duration":
        st.session_state.planned_stop = {"expected_stop_s": float(expected_stop_s)}
        st.session_state.synthetic_train_actual_stop_s = float(expected_stop_s)
    else:
        st.session_state.planned_stop = {"expected_stop_s": float(expected_stop_s)}
        st.session_state.synthetic_train_actual_stop_s = float(actual_stop_s)

    st.session_state.synthetic_include_taxi = bool(include_taxi)
    st.session_state.synthetic_taxi_behavior = str(taxi_behavior)

    st.session_state.vision_params = {
        "crossing_occupancy_threshold": float(crossing_occupancy_threshold),
        "stuck_dwell_threshold_s": float(stuck_dwell_threshold_s),
        "planned_stop_tolerance_s": float(planned_stop_tolerance_s),
        "train_area_threshold": float(train_area_threshold),
        "stationary_speed_px_s": float(stationary_speed_px_s),
        "match_distance_px": float(match_distance_px),
    }

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
        include_taxi = st.toggle("Synthetic crossing includes taxi", value=True)
        taxi_behavior = st.selectbox("Taxi behavior (synthetic)", ["passing", "stuck"], index=0)
        planned_stop_mode = st.selectbox(
            "Planned stop mode (train suppression, demo)",
            ["None", "Planned stop within expected duration", "Planned stop exceeded expected duration"],
            index=0,
        )
        expected_stop_s = st.slider("Expected planned stop duration (s)", 10, 180, 60, 5)
        actual_stop_s = st.slider("Actual stop duration (s) if exceeded (s)", 10, 240, 120, 5)

        stuck_dwell_threshold_s = st.slider("Stuck dwell threshold (s)", 8, 90, 20, 2)
        crossing_occupancy_threshold = st.slider("Crossing occupancy threshold", 0.05, 0.8, 0.25, 0.05)
        planned_stop_tolerance_s = st.slider("Planned-stop tolerance (s)", 0, 60, 10, 2)

        track_id = st.selectbox("Track segment", ["BKS-TIMUR-A", "BKS-TIMUR-B", "BKS-TIMUR-C"])
        min_blob_area = st.slider("Minimum blob area", 200, 6000, 1200, 100)
        conf_threshold = st.slider("Detection confidence threshold", 0.50, 0.95, 0.70, 0.01)
        baseline_latency = st.slider("Legacy signalling latency (s)", 30, 240, 120, 10)
        ai_latency = st.slider("AI-assisted command latency (s)", 2, 45, 8, 1)
        duration_s = st.slider("Run duration per session (s)", 5, 120, 30, 1)
        max_fps = st.slider("Processing FPS cap", 5, 30, 12, 1)
        start = st.button("Start real-time run", width="stretch")
        stop = st.button("Stop real-time run", width="stretch", type="secondary")
        clear_bus = st.button("Clear message bus history", width="stretch")
        if start:
            # Convert "passing/stuck" into synthetic taxi behavior labels.
            taxi_behavior_label = "stuck" if taxi_behavior == "stuck" else "passing"
            start_runner(
                source=source,
                uploaded=uploaded,
                include_taxi=include_taxi,
                taxi_behavior=taxi_behavior_label,
                planned_stop_mode=planned_stop_mode,
                expected_stop_s=float(expected_stop_s),
                actual_stop_s=float(actual_stop_s),
                stuck_dwell_threshold_s=float(stuck_dwell_threshold_s),
                crossing_occupancy_threshold=float(crossing_occupancy_threshold),
                planned_stop_tolerance_s=float(planned_stop_tolerance_s),
                train_area_threshold=2500.0,
                stationary_speed_px_s=35.0,
                match_distance_px=90.0,
            )
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
                    elapsed_s=float(elapsed),
                    include_taxi=bool(st.session_state.get("synthetic_include_taxi", True)),
                    taxi_behavior=str(st.session_state.get("synthetic_taxi_behavior", "passing")),
                    train_actual_stop_s=float(st.session_state.get("synthetic_train_actual_stop_s", 9999.0)),
                    train_moving_speed_px_s=float(st.session_state.get("synthetic_train_moving_speed_px_s", 120.0)),
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
                            elapsed_s=float(elapsed),
                            include_taxi=bool(st.session_state.get("synthetic_include_taxi", True)),
                            taxi_behavior=str(st.session_state.get("synthetic_taxi_behavior", "passing")),
                            train_actual_stop_s=float(st.session_state.get("synthetic_train_actual_stop_s", 9999.0)),
                            train_moving_speed_px_s=float(st.session_state.get("synthetic_train_moving_speed_px_s", 120.0)),
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
                                elapsed_s=float(elapsed),
                                include_taxi=bool(st.session_state.get("synthetic_include_taxi", True)),
                                taxi_behavior=str(st.session_state.get("synthetic_taxi_behavior", "passing")),
                                train_actual_stop_s=float(st.session_state.get("synthetic_train_actual_stop_s", 9999.0)),
                                train_moving_speed_px_s=float(st.session_state.get("synthetic_train_moving_speed_px_s", 120.0)),
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
                now_ts = time.time()
                last_ts = st.session_state.last_frame_ts
                dt_s = 1.0 / float(max_fps) if last_ts is None else max(1e-3, now_ts - float(last_ts))
                st.session_state.last_frame_ts = now_ts

                tracks, next_track_id = st.session_state.tracks, int(st.session_state.next_track_id)
                vision_params = st.session_state.vision_params
                tracks, next_track_id = update_tracks(
                    detections=detections,
                    tracks=tracks,
                    next_track_id=next_track_id,
                    now_ts=now_ts,
                    dt_s=dt_s,
                    match_distance_px=float(vision_params.get("match_distance_px", 80.0)),
                    stationary_speed_px_s=float(vision_params.get("stationary_speed_px_s", 35.0)),
                )
                st.session_state.tracks = tracks
                st.session_state.next_track_id = next_track_id

                obstacle_tracks, obstacle_detections = obstacle_tracks_from_state(
                    tracks=tracks,
                    crossing_occupancy_threshold=float(vision_params.get("crossing_occupancy_threshold", 0.25)),
                    stuck_dwell_threshold_s=float(vision_params.get("stuck_dwell_threshold_s", 20.0)),
                    planned_stop=st.session_state.planned_stop,
                    train_area_threshold=float(vision_params.get("train_area_threshold", 2500.0)),
                    planned_stop_tolerance_s=float(vision_params.get("planned_stop_tolerance_s", 10.0)),
                )
                risk, eta_s = infer_risk_from_obstacles(obstacle_tracks)
                st.session_state.last_risk = risk
                st.session_state.last_obstacle_detections = obstacle_detections

                if risk == "CRITICAL":
                    st.session_state.critical_events += 1

                # Publish only when there is an obstacle-related risk signal.
                if risk in {"MEDIUM", "HIGH", "CRITICAL"} and obstacle_detections:
                    publish_vision_alert(
                        bus=bus,
                        track_id=track_id,
                        risk=risk,
                        eta_s=eta_s,
                        detections=obstacle_detections,
                    )
                    publish_control_dispatch(
                        bus=bus,
                        track_id=track_id,
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
        elif st.session_state.last_obstacle_detections:
            st.caption("Live frame is idle. Showing latest detection state.")
        else:
            st.caption("Press Start to begin streaming frames.")

        st.metric("Current risk", st.session_state.last_risk)
        st.dataframe(pd.DataFrame(st.session_state.last_obstacle_detections), width="stretch", hide_index=True)
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
