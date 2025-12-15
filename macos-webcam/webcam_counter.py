#!/usr/bin/env python3
"""
CrowdCount Webcam Demo
Standalone doorway people counter using macOS webcam.
Uses YOLO + ByteTrack + Deep ReID for accurate counting.

Usage:
    python webcam_counter.py

Controls:
    q - Quit
    r - Reset counts
    f - Flip direction (swap entry/exit)
    d - Toggle debug visualization
    space - Pause/Resume
"""

import cv2
import numpy as np
import time
import math
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
from enum import Enum
from pathlib import Path

# Try imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("ERROR: ultralytics not installed. Install with: pip install ultralytics")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("WARNING: torch not installed.")

try:
    import torchreid
    TORCHREID_AVAILABLE = True
except ImportError:
    TORCHREID_AVAILABLE = False
    print("WARNING: torchreid not installed. Using fallback ReID. Install with: pip install torchreid")


class CrossingDirection(Enum):
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    NONE = "none"


@dataclass
class Detection:
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]
    confidence: float
    timestamp: float
    track_id: Optional[int] = None
    appearance_features: Optional[np.ndarray] = None

    def __hash__(self):
        return hash((self.center, self.timestamp))


class DeepReIDExtractor:
    """Deep ReID using OSNet."""

    def __init__(self, model_name: str = 'osnet_x1_0', device: str = None):
        if not TORCHREID_AVAILABLE:
            raise ImportError("torchreid required")

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[DeepReID] Loading {model_name} on {self.device}...")

        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=True
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.input_size = (128, 256)
        self.pixel_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.pixel_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.feature_dim = 512
        print(f"[DeepReID] Loaded successfully")

    def _preprocess(self, crop: np.ndarray) -> torch.Tensor:
        img = cv2.resize(crop, self.input_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - self.pixel_mean) / self.pixel_std
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return torch.from_numpy(img).float().to(self.device)

    @torch.no_grad()
    def extract(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
        try:
            h, w = frame.shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))

            if x2 <= x1 or y2 <= y1 or (x2 - x1) < 20 or (y2 - y1) < 40:
                return None

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            input_tensor = self._preprocess(crop)
            features = self.model(input_tensor)
            features = features.cpu().numpy().flatten()
            norm = np.linalg.norm(features)
            return features / norm if norm > 0 else features
        except:
            return None


class FallbackReIDExtractor:
    """Fallback HSV histogram ReID."""

    def __init__(self):
        self.feature_dim = 96
        print("[FallbackReID] Using HSV histogram features")

    def extract(self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> Optional[np.ndarray]:
        try:
            h, w = frame.shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))

            if x2 <= x1 or y2 <= y1:
                return None

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return None

            person_resized = cv2.resize(crop, (64, 128))
            hsv = cv2.cvtColor(person_resized, cv2.COLOR_BGR2HSV)

            h_split = 64
            upper, lower = hsv[:h_split, :], hsv[h_split:, :]

            features = []
            for region in [upper, lower]:
                h_hist = cv2.calcHist([region], [0], None, [16], [0, 180]).flatten()
                s_hist = cv2.calcHist([region], [1], None, [8], [0, 256]).flatten()
                v_hist = cv2.calcHist([region], [2], None, [8], [0, 256]).flatten()
                h_hist = h_hist / (h_hist.sum() + 1e-6)
                s_hist = s_hist / (s_hist.sum() + 1e-6)
                v_hist = v_hist / (v_hist.sum() + 1e-6)
                features.extend([h_hist, s_hist, v_hist])

            feature_vector = np.concatenate(features)
            norm = np.linalg.norm(feature_vector)
            return feature_vector / norm if norm > 0 else feature_vector
        except:
            return None


class PersonTracker:
    """Tracks a single person across frames."""

    def __init__(self, track_id: int, initial_detection: Detection):
        self.track_id = track_id
        self.positions = deque(maxlen=60)
        self.velocities = deque(maxlen=15)
        self.creation_time = time.time()
        self.last_update_time = time.time()

        self.crossing_events = []
        self.last_side = None
        self.current_side = None
        self.last_crossing_time = 0
        self.min_crossing_interval = 0.5

        self.side_history = deque(maxlen=10)
        self.detection_count = 0
        self.min_detections_to_count = 3

        self.appearance_history = deque(maxlen=10)
        self.mean_appearance = None

        self.add_detection(initial_detection)

    def add_detection(self, detection: Detection):
        self.detection_count += 1
        self.positions.append(detection.center)
        self.last_update_time = detection.timestamp

        if len(self.positions) >= 2:
            prev = self.positions[-2]
            curr = self.positions[-1]
            dt = max(0.033, detection.timestamp - (self.last_update_time - 0.033))
            self.velocities.append(((curr[0] - prev[0]) / dt, (curr[1] - prev[1]) / dt))

        if detection.appearance_features is not None:
            self.appearance_history.append(detection.appearance_features)
            self.mean_appearance = np.mean(list(self.appearance_history), axis=0)

    def get_appearance_similarity(self, features: np.ndarray) -> float:
        if self.mean_appearance is None or features is None:
            return 0.0
        dot_product = np.dot(self.mean_appearance, features)
        norm_a, norm_b = np.linalg.norm(self.mean_appearance), np.linalg.norm(features)
        return dot_product / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0

    def time_since_update(self) -> float:
        return time.time() - self.last_update_time

    def check_crossing(self, center_x: float, doorway_width: float) -> CrossingDirection:
        if len(self.positions) < 2:
            return CrossingDirection.NONE

        current_x = self.positions[-1][0]
        half_width = doorway_width / 2

        if current_x < (center_x - half_width):
            new_side = 'left'
        elif current_x > (center_x + half_width):
            new_side = 'right'
        else:
            return CrossingDirection.NONE

        self.side_history.append(new_side)
        self.current_side = new_side

        if self.last_side is None:
            self.last_side = new_side
            return CrossingDirection.NONE

        if new_side != self.last_side:
            current_time = time.time()
            if current_time - self.last_crossing_time < self.min_crossing_interval:
                return CrossingDirection.NONE

            if self.detection_count < self.min_detections_to_count:
                self.last_side = new_side
                return CrossingDirection.NONE

            if not self._validate_crossing():
                self.last_side = new_side
                return CrossingDirection.NONE

            direction = (CrossingDirection.LEFT_TO_RIGHT if self.last_side == 'left'
                        else CrossingDirection.RIGHT_TO_LEFT)

            self.last_crossing_time = current_time
            self.crossing_events.append((current_time, direction))
            self.last_side = new_side
            return direction

        return CrossingDirection.NONE

    def _validate_crossing(self) -> bool:
        if len(self.positions) < 3:
            return True

        recent = list(self.positions)[-5:]
        if len(recent) < 2:
            return True

        movements = [recent[i+1][0] - recent[i][0] for i in range(len(recent)-1)]
        movements = [m for m in movements if abs(m) > 2]

        if not movements:
            return False

        positive = sum(1 for m in movements if m > 0)
        negative = len(movements) - positive
        return max(positive, negative) / len(movements) >= 0.5


class WebcamPeopleCounter:
    """Main people counter class for webcam demo."""

    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index
        self.cap = None
        self.model = None
        self.reid_extractor = None

        # Config
        self.config = {
            'doorway_center': 0.5,
            'doorway_width': 0.02,
            'approach_zone': 0.15,
            'max_tracking_distance': 100,
            'tracker_timeout': 3.0,
            'yolo_conf_threshold': 0.25,
            'yolo_iou_threshold': 0.3,
            'min_crossing_interval': 0.5,
            'reid_similarity_threshold': 0.6,
            'reid_model': 'osnet_x1_0',
        }

        # State
        self.trackers: Dict[int, PersonTracker] = {}
        self.next_track_id = 1
        self.crossings = {
            CrossingDirection.LEFT_TO_RIGHT: 0,
            CrossingDirection.RIGHT_TO_LEFT: 0
        }
        self.direction_flipped = False
        self.debug_mode = True
        self.paused = False
        self.device = 'cpu'

    def initialize(self):
        """Initialize camera, YOLO, and ReID."""
        print("Initializing webcam...")
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self.camera_index}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera resolution: {actual_w}x{actual_h}")

        print("Loading YOLO model...")
        if not YOLO_AVAILABLE:
            raise ImportError("ultralytics required")

        self.device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")

        self.model = YOLO('yolo11s.pt')
        self.model.to(self.device)
        print("YOLO loaded")

        print("Loading ReID model...")
        if TORCHREID_AVAILABLE:
            try:
                self.reid_extractor = DeepReIDExtractor(
                    model_name=self.config['reid_model'],
                    device=self.device
                )
            except Exception as e:
                print(f"Deep ReID failed: {e}, using fallback")
                self.reid_extractor = FallbackReIDExtractor()
        else:
            self.reid_extractor = FallbackReIDExtractor()

        print("Initialization complete!")

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Detection]]:
        """Run YOLO detection with ByteTrack on frame."""
        detections = []
        current_time = time.time()

        # Use local bytetrack.yaml if it exists, otherwise use default
        tracker_config = Path(__file__).parent / "bytetrack.yaml"
        if not tracker_config.exists():
            tracker_config = "bytetrack.yaml"

        results = self.model.track(
            frame,
            persist=True,
            tracker=str(tracker_config),
            conf=self.config['yolo_conf_threshold'],
            iou=self.config['yolo_iou_threshold'],
            classes=[0],
            verbose=False
        )

        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                for i in range(len(boxes)):
                    box = boxes[i]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    track_id = int(box.id[0]) if box.id is not None else None

                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    appearance = None
                    if self.reid_extractor:
                        appearance = self.reid_extractor.extract(
                            frame, int(x1), int(y1), int(x2), int(y2)
                        )

                    detection = Detection(
                        center=(cx, cy),
                        bbox=(int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                        confidence=conf,
                        timestamp=current_time,
                        track_id=track_id,
                        appearance_features=appearance
                    )
                    detections.append(detection)

        return frame, detections

    def update_trackers(self, detections: List[Detection], frame_width: int):
        """Update trackers and check for crossings."""
        center_x = frame_width * self.config['doorway_center']
        doorway_width = frame_width * self.config['doorway_width']

        matched_tracker_ids = set()

        for detection in detections:
            tracker_id = None

            # Try ByteTrack ID first
            if detection.track_id is not None and detection.track_id in self.trackers:
                tracker_id = detection.track_id
            else:
                # Try ReID matching
                if detection.appearance_features is not None:
                    best_sim, best_id = 0, None
                    for tid, tracker in self.trackers.items():
                        if tid in matched_tracker_ids:
                            continue
                        sim = tracker.get_appearance_similarity(detection.appearance_features)
                        if sim > best_sim and sim > self.config['reid_similarity_threshold']:
                            best_sim, best_id = sim, tid
                    if best_id is not None:
                        tracker_id = best_id

                # Try distance matching
                if tracker_id is None:
                    best_dist, best_id = float('inf'), None
                    for tid, tracker in self.trackers.items():
                        if tid in matched_tracker_ids or not tracker.positions:
                            continue
                        last_pos = tracker.positions[-1]
                        dist = math.sqrt((detection.center[0] - last_pos[0])**2 +
                                        (detection.center[1] - last_pos[1])**2)
                        if dist < best_dist and dist < self.config['max_tracking_distance']:
                            best_dist, best_id = dist, tid
                    if best_id is not None:
                        tracker_id = best_id

            # Create new tracker if needed
            if tracker_id is None:
                tracker_id = detection.track_id if detection.track_id else self.next_track_id
                self.next_track_id = max(self.next_track_id, tracker_id + 1)
                self.trackers[tracker_id] = PersonTracker(tracker_id, detection)
            else:
                self.trackers[tracker_id].add_detection(detection)

            matched_tracker_ids.add(tracker_id)

            # Check for crossing
            crossing = self.trackers[tracker_id].check_crossing(center_x, doorway_width)
            if crossing != CrossingDirection.NONE:
                self._handle_crossing(crossing)

        # Remove stale trackers
        stale = [tid for tid, t in self.trackers.items()
                 if t.time_since_update() > self.config['tracker_timeout']]
        for tid in stale:
            del self.trackers[tid]

    def _handle_crossing(self, direction: CrossingDirection):
        """Handle a crossing event."""
        if self.direction_flipped:
            if direction == CrossingDirection.LEFT_TO_RIGHT:
                self.crossings[CrossingDirection.RIGHT_TO_LEFT] += 1
                print(f"EXIT detected (flipped) - Total: {self.get_counts()}")
            else:
                self.crossings[CrossingDirection.LEFT_TO_RIGHT] += 1
                print(f"ENTRY detected (flipped) - Total: {self.get_counts()}")
        else:
            self.crossings[direction] += 1
            label = "ENTRY" if direction == CrossingDirection.LEFT_TO_RIGHT else "EXIT"
            print(f"{label} detected - Total: {self.get_counts()}")

    def get_counts(self) -> dict:
        """Get current counts."""
        entries = self.crossings[CrossingDirection.LEFT_TO_RIGHT]
        exits = self.crossings[CrossingDirection.RIGHT_TO_LEFT]
        return {'entries': entries, 'exits': exits, 'inside': entries - exits}

    def draw_visualization(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw visualization overlay on frame."""
        h, w = frame.shape[:2]
        center_x = int(w * self.config['doorway_center'])
        doorway_half = int(w * self.config['doorway_width'] / 2)
        approach = int(w * self.config['approach_zone'])

        # Draw approach zones
        cv2.rectangle(frame, (center_x - doorway_half - approach, 0),
                     (center_x - doorway_half, h), (255, 200, 100), -1)
        cv2.rectangle(frame, (center_x + doorway_half, 0),
                     (center_x + doorway_half + approach, h), (100, 200, 255), -1)

        # Blend zones
        frame_copy = frame.copy()
        cv2.addWeighted(frame_copy, 0.3, frame, 0.7, 0, frame)

        # Draw doorway line
        cv2.line(frame, (center_x, 0), (center_x, h), (0, 255, 0), 3)

        # Draw detections and tracks
        for detection in detections:
            x, y, bw, bh = detection.bbox
            cx, cy = detection.center

            # Color based on side
            if cx < center_x - doorway_half:
                color = (255, 150, 0)  # Blue-ish (left)
            elif cx > center_x + doorway_half:
                color = (0, 150, 255)  # Orange-ish (right)
            else:
                color = (0, 255, 0)  # Green (on line)

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

            # Draw track ID
            if detection.track_id:
                cv2.putText(frame, f"ID:{detection.track_id}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw center point
            cv2.circle(frame, (cx, cy), 5, color, -1)

        # Draw tracker trails
        if self.debug_mode:
            for tracker in self.trackers.values():
                if len(tracker.positions) > 1:
                    pts = list(tracker.positions)
                    for i in range(1, len(pts)):
                        alpha = i / len(pts)
                        color = (int(255 * alpha), int(100 * alpha), 0)
                        cv2.line(frame, pts[i-1], pts[i], color, 2)

        # Draw counts
        counts = self.get_counts()
        cv2.rectangle(frame, (10, 10), (250, 130), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (250, 130), (255, 255, 255), 2)

        cv2.putText(frame, f"Entries: {counts['entries']}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 100), 2)
        cv2.putText(frame, f"Exits: {counts['exits']}", (20, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
        cv2.putText(frame, f"Inside: {counts['inside']}", (20, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw status
        status = "PAUSED" if self.paused else "RUNNING"
        flip_status = "FLIPPED" if self.direction_flipped else "NORMAL"
        cv2.putText(frame, f"{status} | Dir: {flip_status}", (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw help
        cv2.putText(frame, "Q:Quit R:Reset F:Flip D:Debug Space:Pause", (w - 450, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        return frame

    def run(self):
        """Main loop."""
        self.initialize()

        print("\n" + "="*50)
        print("CrowdCount Webcam Demo")
        print("="*50)
        print("Controls:")
        print("  Q - Quit")
        print("  R - Reset counts")
        print("  F - Flip direction")
        print("  D - Toggle debug trails")
        print("  Space - Pause/Resume")
        print("="*50 + "\n")

        fps_time = time.time()
        fps_count = 0
        fps = 0

        try:
            while True:
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to read frame")
                        break

                    # Process frame
                    _, detections = self.process_frame(frame)
                    self.update_trackers(detections, frame.shape[1])

                    # Draw visualization
                    frame = self.draw_visualization(frame, detections)

                    # Calculate FPS
                    fps_count += 1
                    if time.time() - fps_time >= 1.0:
                        fps = fps_count
                        fps_count = 0
                        fps_time = time.time()

                    cv2.putText(frame, f"FPS: {fps}", (frame.shape[1] - 100, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # When paused, just show last frame with PAUSED overlay
                    cv2.putText(frame, "PAUSED", (frame.shape[1]//2 - 80, frame.shape[0]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

                cv2.imshow('CrowdCount Webcam Demo', frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.crossings = {
                        CrossingDirection.LEFT_TO_RIGHT: 0,
                        CrossingDirection.RIGHT_TO_LEFT: 0
                    }
                    print("Counts reset!")
                elif key == ord('f'):
                    self.direction_flipped = not self.direction_flipped
                    print(f"Direction flipped: {self.direction_flipped}")
                elif key == ord('d'):
                    self.debug_mode = not self.debug_mode
                    print(f"Debug mode: {self.debug_mode}")
                elif key == ord(' '):
                    self.paused = not self.paused
                    print(f"{'Paused' if self.paused else 'Resumed'}")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("\n" + "="*50)
            print("Final Counts:")
            counts = self.get_counts()
            print(f"  Entries: {counts['entries']}")
            print(f"  Exits: {counts['exits']}")
            print(f"  Inside: {counts['inside']}")
            print("="*50)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='CrowdCount Webcam Demo')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--no-reid', action='store_true', help='Disable ReID')
    args = parser.parse_args()

    counter = WebcamPeopleCounter(camera_index=args.camera)
    if args.no_reid:
        counter.reid_extractor = None

    counter.run()


if __name__ == '__main__':
    main()
