#!/usr/bin/env python3
"""
Auto-Discovery Overhead Doorway Person Counter
Automatically advertises itself on the network for easy client connection
"""

import cv2
import socket
import pickle
import struct
import numpy as np
import threading
import time
import os
from collections import deque, defaultdict
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum
from ultralytics import YOLO
from zeroconf import ServiceInfo, Zeroconf

class CrossingDirection(Enum):
    LEFT_TO_RIGHT = "left_to_right"
    RIGHT_TO_LEFT = "right_to_left"
    NONE = "none"

@dataclass
class Detection:
    """Represents a single person detection"""
    center: Tuple[int, int]
    bbox: Tuple[int, int, int, int]  # x, y, width, height
    confidence: float
    timestamp: float
    class_id: int
    class_name: str

    def __hash__(self):
        """Make Detection hashable for set operations"""
        return hash((self.center, self.timestamp))

class PersonTracker:
    """Enhanced person tracker with improved crossing detection"""

    def __init__(self, track_id: int, initial_detection: Detection):
        self.track_id = track_id
        self.detections = deque(maxlen=30)
        self.positions = deque(maxlen=30)
        self.velocities = deque(maxlen=10)
        self.creation_time = time.time()
        self.last_update_time = time.time()

        # Crossing state
        self.has_crossed = False
        self.crossing_direction = CrossingDirection.NONE
        self.entry_side = None
        self.exit_side = None
        self.peak_confidence = 0.0

        # Zone tracking
        self.zone_history = deque(maxlen=20)
        self.doorway_crosses = 0
        self.last_doorway_cross_x = None

        # Add initial detection
        self.update(initial_detection)

    def update(self, detection: Detection):
        """Update tracker with new detection"""
        self.detections.append(detection)
        self.positions.append(detection.center)
        self.last_update_time = time.time()
        self.peak_confidence = max(self.peak_confidence, detection.confidence)

        # Calculate velocity
        if len(self.positions) >= 2:
            dx = self.positions[-1][0] - self.positions[-2][0]
            dy = self.positions[-1][1] - self.positions[-2][1]
            dt = detection.timestamp - self.detections[-2].timestamp if len(self.detections) >= 2 else 0.033
            if dt > 0:
                vx = dx / dt
                vy = dy / dt
                self.velocities.append((vx, vy))

    def get_smooth_velocity(self) -> Tuple[float, float]:
        """Get smoothed velocity estimate"""
        if not self.velocities:
            return (0.0, 0.0)

        weights = np.linspace(0.5, 1.0, len(self.velocities))
        vx_avg = np.average([v[0] for v in self.velocities], weights=weights)
        vy_avg = np.average([v[1] for v in self.velocities], weights=weights)

        return (vx_avg, vy_avg)

    def get_trajectory_direction(self) -> CrossingDirection:
        """Determine overall movement direction from trajectory"""
        if len(self.positions) < 5:
            return CrossingDirection.NONE

        positions = list(self.positions)
        x_coords = [p[0] for p in positions]
        indices = list(range(len(positions)))

        if len(set(indices)) > 1:
            slope = np.polyfit(indices, x_coords, 1)[0]

            if slope > 2.0:
                return CrossingDirection.LEFT_TO_RIGHT
            elif slope < -2.0:
                return CrossingDirection.RIGHT_TO_LEFT

        return CrossingDirection.NONE

    def get_path_length(self) -> float:
        """Calculate total path length traveled"""
        if len(self.positions) < 2:
            return 0.0

        total_length = 0.0
        positions = list(self.positions)
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_length += math.sqrt(dx*dx + dy*dy)

        return total_length

    def get_horizontal_displacement(self) -> float:
        """Get net horizontal displacement"""
        if len(self.positions) < 2:
            return 0.0

        return self.positions[-1][0] - self.positions[0][0]

    def is_valid_crossing(self, min_path_length: float, min_displacement: float, min_crosses: int) -> bool:
        """Check if this represents a valid doorway crossing"""
        path_length = self.get_path_length()
        displacement = abs(self.get_horizontal_displacement())

        return (path_length >= min_path_length and
                displacement >= min_displacement and
                self.doorway_crosses >= min_crosses and
                self.peak_confidence >= 0.5)

    def time_since_update(self) -> float:
        """Time since last update"""
        return time.time() - self.last_update_time

class OverheadDoorwayCounter:
    """Enhanced overhead doorway counter with automatic network discovery"""

    def __init__(self, yolo_model_path: str = "combined_yolo11n.pt", port: int = 8485):
        self.port = port
        self.socket = None
        self.client_socket = None
        self.running = False
        self.frame_buffer = deque(maxlen=10)
        self.buffer_lock = threading.Lock()

        # Auto-discovery
        self.zeroconf = None
        self.service_info = None
        self.local_ip = None

        # YOLO model setup
        self.yolo_model_path = yolo_model_path
        try:
            self.model = YOLO(yolo_model_path)
            print(f"✓ Loaded YOLO model: {yolo_model_path}")

            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                print(f"  Classes: {', '.join([f'{k}:{v}' for k, v in self.class_names.items()])}")
            else:
                self.class_names = {0: 'person'}

        except Exception as e:
            print(f"✗ Failed to load YOLO model: {e}")
            raise

        # Tracking
        self.trackers: Dict[int, PersonTracker] = {}
        self.next_track_id = 1
        self.track_id_pool = []

        # Doorway configuration
        self.doorway_config = {
            'doorway_center': 0.5,
            'doorway_width': 0.02,
            'approach_zone': 0.15,
            'min_path_length': 50,
            'min_displacement': 100,
            'max_tracking_distance': 120,
            'tracker_timeout': 2.0,
            'min_doorway_crosses': 1,
            'yolo_conf_threshold': 0.3,
            'yolo_iou_threshold': 0.45
        }

        # Crossing counts
        self.crossings = {
            CrossingDirection.LEFT_TO_RIGHT: 0,
            CrossingDirection.RIGHT_TO_LEFT: 0
        }
        self.crossing_log = []

        # Performance metrics
        self.frame_count = 0
        self.start_time = None
        self.processing_times = deque(maxlen=30)

        # Debug
        self.debug_mode = True
        self.debug_folder = "doorway_debug"
        os.makedirs(self.debug_folder, exist_ok=True)

    def get_local_ip(self) -> str:
        """Get the local IP address"""
        try:
            # Create a socket to determine the local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception as e:
            print(f"Warning: Could not determine local IP, using 0.0.0.0: {e}")
            return "0.0.0.0"

    def register_service(self):
        """Register the service on the network using Zeroconf"""
        try:
            self.local_ip = self.get_local_ip()

            # Create service info
            service_type = "_crowdcount._tcp.local."
            service_name = f"CrowdCount Receiver.{service_type}"

            # Convert IP to bytes
            ip_bytes = socket.inet_aton(self.local_ip)

            self.service_info = ServiceInfo(
                service_type,
                service_name,
                addresses=[ip_bytes],
                port=self.port,
                properties={
                    'version': '1.0',
                    'model': os.path.basename(self.yolo_model_path),
                    'service': 'crowdcount-receiver'
                }
            )

            self.zeroconf = Zeroconf()
            self.zeroconf.register_service(self.service_info)

            print(f"✓ Service registered on network")
            print(f"  Service name: {service_name}")
            print(f"  IP address: {self.local_ip}")
            print(f"  Port: {self.port}")

        except Exception as e:
            print(f"✗ Failed to register service: {e}")
            print(f"  Continuing without auto-discovery...")

    def unregister_service(self):
        """Unregister the service from the network"""
        if self.zeroconf and self.service_info:
            try:
                self.zeroconf.unregister_service(self.service_info)
                self.zeroconf.close()
                print("✓ Service unregistered")
            except Exception as e:
                print(f"Warning: Error unregistering service: {e}")

    def start_server(self) -> bool:
        """Start the server and wait for client connection"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(('0.0.0.0', self.port))
            self.socket.listen(1)

            print(f"✓ Server listening on port {self.port}")
            print(f"  Waiting for client connection...")

            self.client_socket, client_address = self.socket.accept()
            print(f"✓ Client connected from {client_address}")

            return True

        except Exception as e:
            print(f"✗ Server start failed: {e}")
            return False

    def receive_frames(self):
        """Receive frames from camera client"""
        data = b""
        payload_size = struct.calcsize("L")

        while self.running:
            try:
                # Receive message size
                while len(data) < payload_size:
                    packet = self.client_socket.recv(4096)
                    if not packet:
                        self.running = False
                        break
                    data += packet

                if not self.running:
                    break

                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack("L", packed_msg_size)[0]

                # Receive frame data
                while len(data) < msg_size:
                    packet = self.client_socket.recv(min(msg_size - len(data), 65536))
                    if not packet:
                        self.running = False
                        break
                    data += packet

                if not self.running:
                    break

                frame_data = data[:msg_size]
                data = data[msg_size:]

                # Decode frame
                frame = pickle.loads(frame_data)
                frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

                if frame is not None:
                    with self.buffer_lock:
                        self.frame_buffer.append((frame, time.time()))

            except Exception as e:
                print(f"Frame receive error: {e}")
                self.running = False
                break

    def get_zone(self, x_pos: int, frame_width: int) -> str:
        """Determine which zone a position is in"""
        x_ratio = x_pos / frame_width

        center = self.doorway_config['doorway_center']
        half_width = self.doorway_config['doorway_width'] / 2
        approach = self.doorway_config['approach_zone']

        doorway_left = center - half_width
        doorway_right = center + half_width
        left_approach = center - approach
        right_approach = center + approach

        if x_ratio < left_approach:
            return 'far_left'
        elif x_ratio < doorway_left:
            return 'left'
        elif x_ratio <= doorway_right:
            return 'doorway'
        elif x_ratio <= right_approach:
            return 'right'
        else:
            return 'far_right'

    def extract_detections(self, frame: np.ndarray) -> List[Detection]:
        """Extract person detections using YOLO"""
        detections = []
        height, width = frame.shape[:2]
        timestamp = time.time()

        results = self.model(frame,
                           conf=self.doorway_config['yolo_conf_threshold'],
                           iou=self.doorway_config['yolo_iou_threshold'],
                           verbose=False)

        if self.debug_mode and self.frame_count % 100 == 0:
            if not results or not results[0].boxes:
                print(f"  No YOLO detections in frame {self.frame_count}")

        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls)
                    class_name = self.class_names.get(class_id, 'unknown')

                    if class_name == 'person':
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        bbox_width = x2 - x1
                        bbox_height = y2 - y1

                        confidence = float(box.conf)

                        detection = Detection(
                            center=(center_x, center_y),
                            bbox=(x1, y1, bbox_width, bbox_height),
                            confidence=confidence,
                            timestamp=timestamp,
                            class_id=class_id,
                            class_name=class_name
                        )

                        detections.append(detection)

        return detections

    def match_detections_to_trackers(self, detections: List[Detection]) -> Dict[int, Optional[Detection]]:
        """Match detections to existing trackers using Hungarian algorithm"""
        if not self.trackers or not detections:
            return {}

        tracker_ids = list(self.trackers.keys())
        cost_matrix = np.full((len(tracker_ids), len(detections)), 1000.0)

        for i, tid in enumerate(tracker_ids):
            tracker = self.trackers[tid]
            if tracker.positions:
                last_pos = tracker.positions[-1]
                vx, vy = tracker.get_smooth_velocity()

                for j, detection in enumerate(detections):
                    dt = detection.timestamp - tracker.last_update_time
                    predicted_x = last_pos[0] + vx * dt
                    predicted_y = last_pos[1] + vy * dt

                    dx = detection.center[0] - predicted_x
                    dy = detection.center[1] - predicted_y
                    distance = math.sqrt(dx*dx + dy*dy)

                    if distance < self.doorway_config['max_tracking_distance']:
                        cost_matrix[i, j] = distance

        matches = {}
        if cost_matrix.size > 0:
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] < self.doorway_config['max_tracking_distance']:
                    tid = tracker_ids[row]
                    matches[tid] = detections[col]

        return matches

    def update_trackers(self, frame: np.ndarray, detections: List[Detection]):
        """Update trackers with new detections"""
        height, width = frame.shape[:2]

        matches = self.match_detections_to_trackers(detections)
        matched_detections = set(matches.values())

        for tid, detection in matches.items():
            tracker = self.trackers[tid]
            tracker.update(detection)

            zone = self.get_zone(detection.center[0], width)
            tracker.zone_history.append(zone)

            if len(tracker.zone_history) >= 2:
                prev_zone = tracker.zone_history[-2]
                curr_zone = tracker.zone_history[-1]

                if (prev_zone == 'left' and curr_zone in ['doorway', 'right']) or \
                   (prev_zone == 'right' and curr_zone in ['doorway', 'left']) or \
                   (prev_zone == 'doorway' and curr_zone in ['left', 'right']):
                    tracker.doorway_crosses += 1
                    tracker.last_doorway_cross_x = detection.center[0]

            self.check_crossing(tracker, width)

        for detection in detections:
            if detection not in matched_detections:
                if self.track_id_pool:
                    track_id = self.track_id_pool.pop()
                else:
                    track_id = self.next_track_id
                    self.next_track_id += 1

                tracker = PersonTracker(track_id, detection)
                zone = self.get_zone(detection.center[0], width)
                tracker.zone_history.append(zone)
                tracker.entry_side = zone

                self.trackers[track_id] = tracker

        timeout = self.doorway_config['tracker_timeout']
        stale_trackers = []

        for tid, tracker in self.trackers.items():
            if tracker.time_since_update() > timeout:
                if not tracker.has_crossed:
                    self.check_crossing(tracker, width, final_check=True)
                stale_trackers.append(tid)

        for tid in stale_trackers:
            self.track_id_pool.append(tid)
            del self.trackers[tid]

    def check_crossing(self, tracker: PersonTracker, frame_width: int, final_check: bool = False):
        """Check if tracker represents a valid crossing"""
        if tracker.has_crossed or len(tracker.positions) < 5:
            return

        min_path = self.doorway_config['min_path_length']
        min_disp = self.doorway_config['min_displacement']
        min_crosses = self.doorway_config['min_doorway_crosses']

        if not tracker.is_valid_crossing(min_path, min_disp, min_crosses):
            return

        zones = list(tracker.zone_history)
        if not zones:
            return

        start_zone = None
        end_zone = None

        for zone in zones:
            if zone in ['far_left', 'left']:
                start_zone = 'left'
                break
            elif zone in ['far_right', 'right']:
                start_zone = 'right'
                break

        for zone in reversed(zones):
            if zone in ['far_left', 'left']:
                end_zone = 'left'
                break
            elif zone in ['far_right', 'right']:
                end_zone = 'right'
                break

        if start_zone and end_zone and start_zone != end_zone:
            if start_zone == 'left' and end_zone == 'right':
                direction = CrossingDirection.LEFT_TO_RIGHT
            elif start_zone == 'right' and end_zone == 'left':
                direction = CrossingDirection.RIGHT_TO_LEFT
            else:
                return

            if tracker.doorway_crosses > 0:
                self.register_crossing(tracker, direction)

    def register_crossing(self, tracker: PersonTracker, direction: CrossingDirection):
        """Register a valid crossing"""
        tracker.has_crossed = True
        tracker.crossing_direction = direction

        self.crossings[direction] += 1

        crossing_info = {
            'timestamp': time.time(),
            'track_id': tracker.track_id,
            'direction': direction.value,
            'confidence': tracker.peak_confidence,
            'path_length': tracker.get_path_length(),
            'displacement': tracker.get_horizontal_displacement(),
            'duration': tracker.last_update_time - tracker.creation_time
        }
        self.crossing_log.append(crossing_info)

        symbol = "→" if direction == CrossingDirection.LEFT_TO_RIGHT else "←"
        total = sum(self.crossings.values())

        print(f"\n{symbol} CROSSING DETECTED {symbol}")
        print(f"  ID: {tracker.track_id} | Direction: {direction.value}")
        print(f"  Confidence: {tracker.peak_confidence:.2f}")
        print(f"  Path length: {crossing_info['path_length']:.0f}px")
        print(f"  Doorway crosses: {tracker.doorway_crosses}")
        print(f"  L→R: {self.crossings[CrossingDirection.LEFT_TO_RIGHT]} | "
              f"R→L: {self.crossings[CrossingDirection.RIGHT_TO_LEFT]} | "
              f"Total: {total}")

    def draw_visualization(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw debug visualization"""
        vis_frame = frame.copy()
        height, width = frame.shape[:2]

        center_x = int(width * self.doorway_config['doorway_center'])
        doorway_half_width = int(width * self.doorway_config['doorway_width'] / 2)
        approach_width = int(width * self.doorway_config['approach_zone'])

        left_approach_x = center_x - approach_width
        right_approach_x = center_x + approach_width

        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (0, 0), (left_approach_x, height), (255, 200, 200), -1)
        cv2.rectangle(overlay, (right_approach_x, 0), (width, height), (200, 200, 255), -1)
        cv2.addWeighted(overlay, 0.1, vis_frame, 0.9, 0, vis_frame)

        cv2.line(vis_frame, (left_approach_x, 0), (left_approach_x, height), (255, 255, 0), 1)
        cv2.line(vis_frame, (right_approach_x, 0), (right_approach_x, height), (255, 255, 0), 1)

        cv2.line(vis_frame, (center_x - doorway_half_width, 0),
                (center_x - doorway_half_width, height), (0, 255, 0), 3)
        cv2.line(vis_frame, (center_x + doorway_half_width, 0),
                (center_x + doorway_half_width, height), (0, 255, 0), 3)

        cv2.rectangle(vis_frame, (center_x - doorway_half_width, 0),
                     (center_x + doorway_half_width, height), (0, 255, 0), -1)
        cv2.addWeighted(vis_frame, 0.7, vis_frame, 0.3, 0, vis_frame)

        cv2.line(vis_frame, (center_x, 0), (center_x, height), (0, 150, 0), 2)

        cv2.putText(vis_frame, "LEFT ZONE", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
        cv2.putText(vis_frame, "DOORWAY", (center_x - 50, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(vis_frame, "RIGHT ZONE", (width - 180, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        for detection in detections:
            x, y, w, h = detection.bbox
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(vis_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        for tid, tracker in self.trackers.items():
            if not tracker.positions:
                continue

            current_pos = tracker.positions[-1]
            zone = self.get_zone(current_pos[0], width)

            if tracker.has_crossed:
                color = (0, 255, 255)
            elif tracker.doorway_crosses > 0:
                color = (0, 255, 128)
            elif zone == 'doorway':
                color = (0, 255, 0)
            elif zone in ['left', 'far_left']:
                color = (255, 100, 100)
            else:
                color = (100, 100, 255)

            if len(tracker.positions) > 1:
                points = list(tracker.positions)
                for i in range(1, len(points)):
                    cv2.line(vis_frame, points[i-1], points[i], color, 2)

            cv2.circle(vis_frame, current_pos, 8, color, -1)
            cv2.circle(vis_frame, current_pos, 10, (0, 0, 0), 2)

            info_text = f"ID:{tid}"
            if tracker.doorway_crosses > 0:
                info_text += f" X:{tracker.doorway_crosses}"

            cv2.putText(vis_frame, info_text,
                       (current_pos[0] - 20, current_pos[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
            cv2.putText(vis_frame, info_text,
                       (current_pos[0] - 20, current_pos[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            vx, vy = tracker.get_smooth_velocity()
            if abs(vx) > 10:
                arrow_end = (int(current_pos[0] + vx * 0.5), current_pos[1])
                cv2.arrowedLine(vis_frame, current_pos, arrow_end, (255, 255, 0), 3)

        stats_y = height - 120
        cv2.rectangle(vis_frame, (5, stats_y - 5), (400, height - 5), (0, 0, 0), -1)
        cv2.rectangle(vis_frame, (5, stats_y - 5), (400, height - 5), (255, 255, 255), 2)

        total = sum(self.crossings.values())
        net_flow = self.crossings[CrossingDirection.LEFT_TO_RIGHT] - self.crossings[CrossingDirection.RIGHT_TO_LEFT]

        stats_text = [
            f"Active Trackers: {len(self.trackers)}",
            f"L->R: {self.crossings[CrossingDirection.LEFT_TO_RIGHT]}  |  R->L: {self.crossings[CrossingDirection.RIGHT_TO_LEFT]}",
            f"Total Crossings: {total}",
            f"Net Flow (L->R): {net_flow:+d}"
        ]

        for i, text in enumerate(stats_text):
            cv2.putText(vis_frame, text, (15, stats_y + 20 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if self.processing_times:
            avg_time = np.mean(self.processing_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            cv2.putText(vis_frame, f"FPS: {fps:.1f}", (width - 120, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(vis_frame, f"Model: {os.path.basename(self.yolo_model_path)}",
                   (10, height - 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        return vis_frame

    def process_frames(self):
        """Main processing loop"""
        cv2.namedWindow('Overhead Doorway Counter - Auto Discovery', cv2.WINDOW_NORMAL)

        print("\n" + "="*60)
        print("OVERHEAD DOORWAY COUNTER - Started (Auto-Discovery)")
        print("="*60)
        print(f"YOLO Model: {self.yolo_model_path}")
        print("Press 'q' to quit, 's' to save debug frame")
        print("="*60 + "\n")

        while self.running:
            frame_data = None

            with self.buffer_lock:
                if self.frame_buffer:
                    frame_data = self.frame_buffer.popleft()

            if frame_data is not None:
                frame, timestamp = frame_data
                process_start = time.time()

                self.frame_count += 1

                detections = self.extract_detections(frame)

                if self.frame_count % 100 == 0 and detections:
                    print(f"  Detections found: {len(detections)}")

                self.update_trackers(frame, detections)

                vis_frame = self.draw_visualization(frame, detections)

                cv2.imshow('Overhead Doorway Counter - Auto Discovery', vis_frame)

                process_time = time.time() - process_start
                self.processing_times.append(process_time)

                if self.frame_count % 300 == 0:
                    elapsed = time.time() - self.start_time
                    avg_fps = self.frame_count / elapsed
                    total = sum(self.crossings.values())

                    print(f"Frame {self.frame_count} | "
                          f"FPS: {avg_fps:.1f} | "
                          f"Active: {len(self.trackers)} | "
                          f"Total: {total}")

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"{self.debug_folder}/debug_{self.frame_count}_{int(time.time())}.jpg"
                    cv2.imwrite(filename, vis_frame)
                    print(f"Saved debug frame: {filename}")

            else:
                time.sleep(0.001)

        cv2.destroyAllWindows()

    def start(self):
        """Start the doorway counter with auto-discovery"""
        # Register service on network
        self.register_service()

        # Start server
        if not self.start_server():
            self.unregister_service()
            return False

        self.running = True
        self.start_time = time.time()

        # Start receiver thread
        receiver_thread = threading.Thread(target=self.receive_frames)
        receiver_thread.daemon = True
        receiver_thread.start()

        # Process frames in main thread
        try:
            self.process_frames()
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

        return True

    def cleanup(self):
        """Clean up resources and print summary"""
        self.running = False

        # Unregister service
        self.unregister_service()

        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass

        if self.socket:
            try:
                self.socket.close()
            except:
                pass

        cv2.destroyAllWindows()

        if self.start_time:
            elapsed = time.time() - self.start_time
            total = sum(self.crossings.values())

            print("\n" + "="*60)
            print("OVERHEAD DOORWAY COUNTER - Summary")
            print("="*60)
            print(f"Runtime: {elapsed:.1f} seconds")
            print(f"Frames processed: {self.frame_count:,}")
            print(f"Average FPS: {self.frame_count/elapsed:.1f}")
            print(f"Model: {self.yolo_model_path}")
            print(f"\nCrossing Counts:")
            print(f"  Left → Right: {self.crossings[CrossingDirection.LEFT_TO_RIGHT]}")
            print(f"  Right → Left: {self.crossings[CrossingDirection.RIGHT_TO_LEFT]}")
            print(f"  Total: {total}")

            if total > 0:
                net_flow = self.crossings[CrossingDirection.LEFT_TO_RIGHT] - self.crossings[CrossingDirection.RIGHT_TO_LEFT]
                print(f"  Net Flow (L→R): {net_flow:+d}")

                if self.crossing_log:
                    avg_confidence = np.mean([c['confidence'] for c in self.crossing_log])
                    avg_duration = np.mean([c['duration'] for c in self.crossing_log])
                    print(f"\nAverage Metrics:")
                    print(f"  Detection confidence: {avg_confidence:.2f}")
                    print(f"  Crossing duration: {avg_duration:.2f}s")

            print("="*60)

            if self.crossing_log:
                import json
                log_file = f"{self.debug_folder}/crossing_log_{int(time.time())}.json"
                with open(log_file, 'w') as f:
                    json.dump(self.crossing_log, f, indent=2)
                print(f"\nCrossing log saved to: {log_file}")

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='Auto-Discovery Overhead Doorway Person Counter')
    parser.add_argument('--model', type=str, default='combined_yolo11n.pt',
                        help='Path to YOLO model file (default: combined_yolo11n.pt)')
    parser.add_argument('--port', type=int, default=8485,
                        help='Port number (default: 8485)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='YOLO confidence threshold (default: 0.3)')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='YOLO IOU threshold for NMS (default: 0.45)')

    args = parser.parse_args()

    print("="*60)
    print("CrowdCount Auto-Discovery Receiver")
    print("="*60)
    print("This receiver will automatically advertise itself on the network.")
    print("Camera clients can discover and connect without manual IP config.")
    print("="*60 + "\n")

    counter = OverheadDoorwayCounter(args.model, args.port)
    counter.debug_mode = args.debug
    counter.doorway_config['yolo_conf_threshold'] = args.conf
    counter.doorway_config['yolo_iou_threshold'] = args.iou

    try:
        counter.start()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        counter.cleanup()

if __name__ == "__main__":
    main()
