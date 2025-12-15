#!/usr/bin/env python3
"""
Enhanced Overhead Doorway Person Counter
Using custom YOLO model for detection and tracking
Modified to allow multiple crossings per person while in frame
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
import requests
import json

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
        self.detections = deque(maxlen=30)  # Keep more history
        self.positions = deque(maxlen=30)
        self.velocities = deque(maxlen=10)
        self.creation_time = time.time()
        self.last_update_time = time.time()
        
        # Multiple crossing state tracking
        self.total_crossings = 0
        self.crossing_events = []  # List of (timestamp, direction) tuples
        self.last_side = None  # Track which side of center line person is on
        self.current_side = None
        self.crossing_in_progress = False
        self.last_crossing_time = 0
        self.min_crossing_interval = 0.3  # Minimum seconds between crossings to avoid double counting
        
        # Overall tracking stats
        self.entry_side = None
        self.exit_side = None
        self.peak_confidence = 0.0
        
        # Zone tracking
        self.zone_history = deque(maxlen=20)
        self.doorway_crosses = 0  # Count times crossed doorway line
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
    
    def check_crossing(self, frame_width: int, doorway_center: float, doorway_width: float) -> Optional[CrossingDirection]:
        """
        Check if a crossing has occurred and return the direction.
        This method can detect multiple crossings while the person remains in frame.
        """
        if len(self.positions) < 2:
            return None
            
        current_x = self.positions[-1][0]
        center_x = frame_width * doorway_center
        half_width = frame_width * doorway_width / 2
        
        # Determine which side of the center line the person is on
        if current_x < (center_x - half_width):
            self.current_side = 'left'
        elif current_x > (center_x + half_width):
            self.current_side = 'right'
        else:
            # Person is exactly on the line
            return None
        
        # Check if we have a previous side recorded
        if self.last_side is None:
            self.last_side = self.current_side
            return None
        
        # Check for crossing
        crossing_direction = None
        current_time = time.time()
        
        # Crossing from left to right
        if self.last_side == 'left' and self.current_side == 'right':
            # Check if enough time has passed since last crossing to avoid double counting
            if current_time - self.last_crossing_time >= self.min_crossing_interval:
                crossing_direction = CrossingDirection.LEFT_TO_RIGHT
                self.total_crossings += 1
                self.crossing_events.append((current_time, CrossingDirection.LEFT_TO_RIGHT))
                self.last_crossing_time = current_time
                self.doorway_crosses += 1
                
        # Crossing from right to left
        elif self.last_side == 'right' and self.current_side == 'left':
            if current_time - self.last_crossing_time >= self.min_crossing_interval:
                crossing_direction = CrossingDirection.RIGHT_TO_LEFT
                self.total_crossings += 1
                self.crossing_events.append((current_time, CrossingDirection.RIGHT_TO_LEFT))
                self.last_crossing_time = current_time
                self.doorway_crosses += 1
        
        # Update last side
        if self.current_side in ['left', 'right']:
            self.last_side = self.current_side
        
        return crossing_direction
    
    def get_smooth_velocity(self) -> Tuple[float, float]:
        """Get smoothed velocity estimate"""
        if not self.velocities:
            return (0.0, 0.0)
        
        # Weighted average with recent velocities having more weight
        weights = np.linspace(0.5, 1.0, len(self.velocities))
        vx_avg = np.average([v[0] for v in self.velocities], weights=weights)
        vy_avg = np.average([v[1] for v in self.velocities], weights=weights)
        
        return (vx_avg, vy_avg)
    
    def get_trajectory_direction(self) -> CrossingDirection:
        """Determine overall movement direction from trajectory"""
        if len(self.positions) < 5:
            return CrossingDirection.NONE
        
        # Use linear regression for robust direction estimation
        positions = list(self.positions)
        x_coords = [p[0] for p in positions]
        indices = list(range(len(positions)))
        
        # Calculate slope of movement
        if len(set(indices)) > 1:  # Avoid division by zero
            slope = np.polyfit(indices, x_coords, 1)[0]
            
            # Significant rightward movement
            if slope > 2.0:
                return CrossingDirection.LEFT_TO_RIGHT
            # Significant leftward movement
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
    
    def is_valid_tracker(self, min_path_length: float, min_confidence: float = 0.5) -> bool:
        """Check if this tracker represents a valid person (for filtering noise)"""
        path_length = self.get_path_length()
        
        return (path_length >= min_path_length and 
                self.peak_confidence >= min_confidence)
    
    def time_since_update(self) -> float:
        """Time since last update"""
        return time.time() - self.last_update_time

class OverheadDoorwayCounter:
    """Enhanced overhead doorway counter using YOLO"""
    
    def __init__(self, pi_ip: str, yolo_model_path: str = "combined_yolo11n.pt", port: int = 8485):
        self.pi_ip = pi_ip
        self.port = port
        self.socket = None
        self.running = False
        self.frame_buffer = deque(maxlen=10)
        self.buffer_lock = threading.Lock()
        
        # API Configuration
        self.api_config = {
            'enabled': True,  # Set to False to disable API calls
            'base_url': 'https://api.crowdcount.tech',
            'user_id': 'Demo',
            'password': 'password',
            'timeout': 5  # API call timeout in seconds
        }
        
        # YOLO model setup
        self.yolo_model_path = yolo_model_path
        try:
            self.model = YOLO(yolo_model_path)
            print(f"✓ Loaded YOLO model: {yolo_model_path}")
            
            # Get class names from model
            if hasattr(self.model, 'names'):
                self.class_names = self.model.names
                print(f"  Classes: {', '.join([f'{k}:{v}' for k, v in self.class_names.items()])}")
            else:
                self.class_names = {0: 'person'}  # Default to person class
                
        except Exception as e:
            print(f"✗ Failed to load YOLO model: {e}")
            raise
        
        # Tracking
        self.trackers: Dict[int, PersonTracker] = {}
        self.next_track_id = 1
        self.track_id_pool = []  # Reuse old IDs
        
        # Doorway configuration
        self.doorway_config = {
            'doorway_center': 0.5,      # 50% of frame width (center line)
            'doorway_width': 0.02,      # 2% of frame width (very narrow line)
            'approach_zone': 0.15,      # 15% approach zone on each side
            'min_path_length': 50,      # Minimum pixels traveled for valid tracker
            'max_tracking_distance': 120,  # Maximum distance for matching
            'tracker_timeout': 2.0,     # Seconds before removing stale tracker
            'yolo_conf_threshold': 0.3, # YOLO confidence threshold (lowered for head detection)
            'yolo_iou_threshold': 0.45  # YOLO IOU threshold for NMS
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
        
        # Debug mode
        self.debug_mode = False
        self.debug_folder = "debug_frames"
        if not os.path.exists(self.debug_folder):
            os.makedirs(self.debug_folder)
    
    def connect_to_pi(self) -> bool:
        """Connect to Raspberry Pi camera server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.pi_ip, self.port))
            print(f"✓ Connected to Raspberry Pi at {self.pi_ip}:{self.port}")
            return True
        except Exception as e:
            print(f"✗ Failed to connect to {self.pi_ip}:{self.port} - {e}")
            return False
    
    def api_add_to_count(self, count: int = 1) -> bool:
        """
        Make API call to add to count when person goes RIGHT (entering).
        This is called for LEFT_TO_RIGHT crossings.
        
        Args:
            count: Number to add to the count (default 1)
            
        Returns:
            True if API call was successful, False otherwise
        """
        if not self.api_config['enabled']:
            return True  # Return success if API calls are disabled
            
        try:
            url = f"{self.api_config['base_url']}/add_to_count"
            data = {
                'user_id': self.api_config['user_id'],
                'password': self.api_config['password'],
                'count_to_add': str(count)
            }
            
            response = requests.post(
                url, 
                data=data,
                timeout=self.api_config['timeout']
            )
            
            if response.status_code == 200:
                print(f"  ✓ API: Added {count} to remote count")
                return True
            else:
                print(f"  ✗ API Error: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"  ✗ API Timeout: Call took longer than {self.api_config['timeout']}s")
            return False
        except requests.exceptions.ConnectionError:
            print(f"  ✗ API Connection Error: Could not reach {self.api_config['base_url']}")
            return False
        except Exception as e:
            print(f"  ✗ API Error: {e}")
            return False
    
    def api_sub_from_count(self, count: int = 1) -> bool:
        """
        Make API call to subtract from count when person goes LEFT (exiting).
        This is called for RIGHT_TO_LEFT crossings.
        
        Args:
            count: Number to subtract from the count (default 1)
            
        Returns:
            True if API call was successful, False otherwise
        """
        if not self.api_config['enabled']:
            return True  # Return success if API calls are disabled
            
        try:
            url = f"{self.api_config['base_url']}/sub_from_count"
            data = {
                'user_id': self.api_config['user_id'],
                'password': self.api_config['password'],
                'count_to_sub': str(count)
            }
            
            response = requests.post(
                url,
                data=data,
                timeout=self.api_config['timeout']
            )
            
            if response.status_code == 200:
                print(f"  ✓ API: Subtracted {count} from remote count")
                return True
            else:
                print(f"  ✗ API Error: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            print(f"  ✗ API Timeout: Call took longer than {self.api_config['timeout']}s")
            return False
        except requests.exceptions.ConnectionError:
            print(f"  ✗ API Connection Error: Could not reach {self.api_config['base_url']}")
            return False
        except Exception as e:
            print(f"  ✗ API Error: {e}")
            return False
    
    def receive_frames(self):
        """Receive frames from Raspberry Pi"""
        while self.running:
            try:
                # Read message size
                raw_msglen = self.recvall(4)
                if not raw_msglen:
                    continue
                
                # Try big-endian first (new format), then little-endian (old format)
                try:
                    msglen = struct.unpack('>L', raw_msglen)[0]
                except:
                    msglen = struct.unpack('L', raw_msglen)[0]
                
                # Sanity check on message length (JPEG should be much smaller than raw)
                if msglen > 5000000:  # 5MB max for JPEG frame
                    print(f"Warning: Unreasonable message size {msglen}, skipping")
                    continue
                
                # Read the message data
                data = self.recvall(msglen)
                if not data:
                    continue
                
                # Unpickle the data
                unpacked_data = pickle.loads(data)
                
                # Handle different formats for compatibility
                frame = None
                timestamp = time.time()
                
                if isinstance(unpacked_data, dict):
                    # New format - dictionary with timestamp
                    if 'jpeg' in unpacked_data:
                        # JPEG encoded frame (most efficient)
                        frame = cv2.imdecode(unpacked_data['jpeg'], cv2.IMREAD_COLOR)
                        timestamp = unpacked_data.get('timestamp', time.time())
                    elif 'frame' in unpacked_data:
                        # Raw frame (legacy support)
                        frame = unpacked_data['frame']
                        timestamp = unpacked_data.get('timestamp', time.time())
                else:
                    # Old format - just JPEG encoded frame
                    frame = cv2.imdecode(unpacked_data, cv2.IMREAD_COLOR)
                
                if frame is not None:
                    # Add to buffer
                    with self.buffer_lock:
                        self.frame_buffer.append((frame, timestamp))
                else:
                    print("Warning: Failed to decode frame")
                    
            except struct.error as e:
                if self.running:
                    print(f"Struct unpack error: {e}")
                    continue
            except pickle.UnpicklingError as e:
                if self.running:
                    print(f"Pickle error: {e}")
                    continue
            except Exception as e:
                if self.running:
                    print(f"Receive error: {e}")
                    # Try to reconnect after a brief pause
                    time.sleep(1)
                    break
    
    def recvall(self, n):
        """Helper to receive n bytes or return None if EOF is hit"""
        data = bytearray()
        while len(data) < n:
            packet = self.socket.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return bytes(data)
    
    def extract_detections(self, frame) -> List[Detection]:
        """Extract person detections from frame using YOLO"""
        detections = []
        
        try:
            # Run YOLO inference
            results = self.model(
                frame, 
                conf=self.doorway_config['yolo_conf_threshold'],
                iou=self.doorway_config['yolo_iou_threshold'],
                verbose=False
            )
            
            if results and len(results) > 0:
                result = results[0]
                
                if result.boxes is not None and len(result.boxes) > 0:
                    # Process each detection
                    for box in result.boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Calculate center and dimensions
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        width = int(x2 - x1)
                        height = int(y2 - y1)
                        
                        # Get confidence and class
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.class_names.get(class_id, 'unknown')
                        
                        # Create detection
                        detection = Detection(
                            center=(center_x, center_y),
                            bbox=(int(x1), int(y1), width, height),
                            confidence=confidence,
                            timestamp=time.time(),
                            class_id=class_id,
                            class_name=class_name
                        )
                        
                        detections.append(detection)
                        
        except Exception as e:
            if self.debug_mode:
                print(f"Detection error: {e}")
        
        return detections
    
    def match_detection_to_tracker(self, detection: Detection) -> Optional[int]:
        """Match a detection to existing tracker using simple distance metric"""
        if not self.trackers:
            return None
        
        best_match_id = None
        min_distance = self.doorway_config['max_tracking_distance']
        
        for track_id, tracker in self.trackers.items():
            if len(tracker.positions) == 0:
                continue
            
            # Calculate distance to last known position
            last_pos = tracker.positions[-1]
            dist = math.sqrt(
                (detection.center[0] - last_pos[0])**2 + 
                (detection.center[1] - last_pos[1])**2
            )
            
            # Consider velocity for prediction
            vx, vy = tracker.get_smooth_velocity()
            time_diff = detection.timestamp - tracker.last_update_time
            
            # Predict where tracker should be
            predicted_x = last_pos[0] + vx * time_diff
            predicted_y = last_pos[1] + vy * time_diff
            
            # Distance to predicted position
            pred_dist = math.sqrt(
                (detection.center[0] - predicted_x)**2 + 
                (detection.center[1] - predicted_y)**2
            )
            
            # Use minimum of actual and predicted distance
            final_dist = min(dist, pred_dist)
            
            if final_dist < min_distance:
                min_distance = final_dist
                best_match_id = track_id
        
        return best_match_id
    
    def update_trackers(self, frame, detections: List[Detection]):
        """Update trackers with new detections and check for crossings"""
        height, width = frame.shape[:2]
        current_time = time.time()
        
        # Match detections to existing trackers
        matched_trackers = set()
        unmatched_detections = []
        
        for detection in detections:
            track_id = self.match_detection_to_tracker(detection)
            
            if track_id is not None:
                # Update existing tracker
                self.trackers[track_id].update(detection)
                matched_trackers.add(track_id)
                
                # Check for crossing with this updated position
                crossing_dir = self.trackers[track_id].check_crossing(
                    width, 
                    self.doorway_config['doorway_center'],
                    self.doorway_config['doorway_width']
                )
                
                if crossing_dir and crossing_dir != CrossingDirection.NONE:
                    # A crossing occurred!
                    self.crossings[crossing_dir] += 1
                    
                    # Make API call based on direction
                    # Use threading to make API calls non-blocking
                    if crossing_dir == CrossingDirection.LEFT_TO_RIGHT:
                        # Person going RIGHT (entering)
                        api_thread = threading.Thread(
                            target=self.api_add_to_count,
                            args=(1,),
                            daemon=True
                        )
                        api_thread.start()
                    elif crossing_dir == CrossingDirection.RIGHT_TO_LEFT:
                        # Person going LEFT (exiting)
                        api_thread = threading.Thread(
                            target=self.api_sub_from_count,
                            args=(1,),
                            daemon=True
                        )
                        api_thread.start()
                    
                    # Log the crossing
                    crossing_info = {
                        'timestamp': current_time,
                        'track_id': track_id,
                        'direction': crossing_dir.value,
                        'confidence': self.trackers[track_id].peak_confidence,
                        'duration': current_time - self.trackers[track_id].creation_time,
                        'total_crossings_by_person': self.trackers[track_id].total_crossings,
                        'position': detection.center
                    }
                    self.crossing_log.append(crossing_info)
                    
                    # Print crossing info
                    total = sum(self.crossings.values())
                    api_status = "→ API" if self.api_config['enabled'] else ""
                    print(f"\n{'='*50}")
                    print(f"CROSSING DETECTED #{total} {api_status}")
                    print(f"  Direction: {crossing_dir.value}")
                    print(f"  Track ID: {track_id}")
                    print(f"  Person's crossing #{self.trackers[track_id].total_crossings}")
                    print(f"  Confidence: {self.trackers[track_id].peak_confidence:.2f}")
                    print(f"  L→R: {self.crossings[CrossingDirection.LEFT_TO_RIGHT]} | "
                          f"R→L: {self.crossings[CrossingDirection.RIGHT_TO_LEFT]}")
                    print(f"{'='*50}\n")
            else:
                unmatched_detections.append(detection)
        
        # Create new trackers for unmatched detections
        for detection in unmatched_detections:
            # Get new track ID
            if self.track_id_pool:
                new_id = self.track_id_pool.pop(0)
            else:
                new_id = self.next_track_id
                self.next_track_id += 1
            
            # Create new tracker
            self.trackers[new_id] = PersonTracker(new_id, detection)
            
            if self.debug_mode:
                print(f"New tracker #{new_id} at {detection.center}")
        
        # Remove stale trackers
        trackers_to_remove = []
        for track_id, tracker in self.trackers.items():
            if tracker.time_since_update() > self.doorway_config['tracker_timeout']:
                trackers_to_remove.append(track_id)
                
                # Return ID to pool for reuse
                if track_id not in self.track_id_pool:
                    self.track_id_pool.append(track_id)
                
                if self.debug_mode:
                    print(f"Removing stale tracker #{track_id} "
                          f"(crossings: {tracker.total_crossings})")
        
        for track_id in trackers_to_remove:
            del self.trackers[track_id]
    
    def draw_visualization(self, frame, detections: List[Detection]):
        """Draw visualization overlay on frame"""
        vis_frame = frame.copy()
        height, width = vis_frame.shape[:2]
        
        # Calculate doorway line position
        center_x = int(width * self.doorway_config['doorway_center'])
        line_half_width = int(width * self.doorway_config['doorway_width'] / 2)
        
        # Draw approach zones
        approach_width = int(width * self.doorway_config['approach_zone'])
        
        # Left approach zone
        cv2.rectangle(vis_frame,
                     (center_x - approach_width - line_half_width, 0),
                     (center_x - line_half_width, height),
                     (100, 100, 200), 1)
        
        # Right approach zone
        cv2.rectangle(vis_frame,
                     (center_x + line_half_width, 0),
                     (center_x + approach_width + line_half_width, height),
                     (100, 100, 200), 1)
        
        # Draw doorway center line (bright green)
        cv2.line(vis_frame, 
                (center_x, 0), 
                (center_x, height), 
                (0, 255, 0), 2)
        
        # Draw thin lines on either side of center to show doorway width
        cv2.line(vis_frame,
                (center_x - line_half_width, 0),
                (center_x - line_half_width, height),
                (0, 180, 0), 1)
        cv2.line(vis_frame,
                (center_x + line_half_width, 0),
                (center_x + line_half_width, height),
                (0, 180, 0), 1)
        
        # Draw current detections
        for detection in detections:
            x, y, w, h = detection.bbox
            
            # Draw bounding box
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            
            # Draw center point
            cv2.circle(vis_frame, detection.center, 5, (0, 255, 255), -1)
            
            # Draw confidence
            conf_text = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(vis_frame, conf_text, 
                       (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw trackers
        for track_id, tracker in self.trackers.items():
            if len(tracker.positions) < 2:
                continue
            
            # Determine color based on side
            if tracker.current_side == 'left':
                track_color = (255, 100, 100)  # Blue-ish
            elif tracker.current_side == 'right':
                track_color = (100, 100, 255)  # Red-ish
            else:
                track_color = (0, 255, 0)  # Green for on the line
            
            # Draw path
            positions = list(tracker.positions)
            for i in range(1, len(positions)):
                pt1 = (int(positions[i-1][0]), int(positions[i-1][1]))
                pt2 = (int(positions[i][0]), int(positions[i][1]))
                thickness = max(1, int(3 * (i / len(positions))))
                cv2.line(vis_frame, pt1, pt2, track_color, thickness)
            
            # Draw current position
            if positions:
                current_pos = positions[-1]
                cv2.circle(vis_frame, 
                          (int(current_pos[0]), int(current_pos[1])), 
                          8, track_color, -1)
                
                # Draw track ID and crossing count
                info_text = f"#{track_id}"
                if tracker.total_crossings > 0:
                    info_text += f" (x{tracker.total_crossings})"
                    
                cv2.putText(vis_frame, info_text,
                           (int(current_pos[0]) + 10, int(current_pos[1]) - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_color, 2)
                
                # Draw velocity arrow
                vx, vy = tracker.get_smooth_velocity()
                if abs(vx) > 5 or abs(vy) > 5:
                    arrow_end = (int(current_pos[0] + vx * 0.5),
                               int(current_pos[1] + vy * 0.5))
                    cv2.arrowedLine(vis_frame,
                                  (int(current_pos[0]), int(current_pos[1])),
                                  arrow_end,
                                  (0, 255, 0), 2, tipLength=0.3)
        
        # Draw statistics overlay
        stats_bg_height = 180
        overlay = vis_frame.copy()
        cv2.rectangle(overlay, (0, 0), (350, stats_bg_height), (0, 0, 0), -1)
        vis_frame = cv2.addWeighted(vis_frame, 0.7, overlay, 0.3, 0)
        
        # Statistics text
        total_crossings = sum(self.crossings.values())
        lr_count = self.crossings[CrossingDirection.LEFT_TO_RIGHT]
        rl_count = self.crossings[CrossingDirection.RIGHT_TO_LEFT]
        net_flow = lr_count - rl_count
        
        cv2.putText(vis_frame, "CROWDCOUNT COUNTER",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(vis_frame, f"Total Crossings: {total_crossings}",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(vis_frame, f"Left -> Right: {lr_count}",
                   (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        cv2.putText(vis_frame, f"Right -> Left: {rl_count}",
                   (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 1)
        cv2.putText(vis_frame, f"Net Flow (L->R): {net_flow:+d}",
                   (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(vis_frame, f"Active Trackers: {len(self.trackers)}",
                   (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Frame info
        if self.start_time:
            elapsed = time.time() - self.start_time
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(vis_frame, f"FPS: {fps:.1f}",
                       (10, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Processing time
        if self.processing_times:
            avg_time = np.mean(self.processing_times) * 1000
            cv2.putText(vis_frame, f"Process: {avg_time:.1f}ms",
                       (10, height - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Instructions
        cv2.putText(vis_frame, "Press 'q' to quit, 's' to save frame",
                   (10, height - 140),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        return vis_frame
    
    def process_frames(self):
        """Main processing loop"""
        cv2.namedWindow('CrowdCount Counter - Demo', cv2.WINDOW_NORMAL)
        
        print("\n" + "="*60)
        print("OVERHEAD DOORWAY COUNTER - Started")
        print("Multi-Crossing Mode Enabled")
        if self.api_config['enabled']:
            print(f"API Integration: {self.api_config['base_url']}")
            print("  L→R crossings will ADD to remote count")
            print("  R→L crossings will SUBTRACT from remote count")
        else:
            print("API Integration: Disabled")
        print("="*60)
        print(f"Connected to: {self.pi_ip}")
        print(f"YOLO Model: {self.yolo_model_path}")
        print("Press 'q' to quit, 's' to save debug frame")
        print("="*60 + "\n")
        
        while self.running:
            frame_data = None
            
            # Get latest frame
            with self.buffer_lock:
                if self.frame_buffer:
                    frame_data = self.frame_buffer.popleft()
            
            if frame_data is not None:
                frame, timestamp = frame_data
                process_start = time.time()
                
                self.frame_count += 1
                
                # Extract detections using YOLO
                detections = self.extract_detections(frame)
                
                # Debug: Print detection count occasionally
                if self.frame_count % 100 == 0 and detections:
                    print(f"  Detections found: {len(detections)}")
                
                # Update trackers
                self.update_trackers(frame, detections)
                
                # Create visualization
                vis_frame = self.draw_visualization(frame, detections)
                
                # Show frame
                cv2.imshow('CrowdCount Counter - Demo', vis_frame)
                
                # Track processing time
                process_time = time.time() - process_start
                self.processing_times.append(process_time)
                
                # Periodic status update
                if self.frame_count % 300 == 0:
                    elapsed = time.time() - self.start_time
                    avg_fps = self.frame_count / elapsed
                    total = sum(self.crossings.values())
                    
                    # Count total crossings by all people
                    total_individual_crossings = sum(
                        tracker.total_crossings for tracker in self.trackers.values()
                    )
                    
                    print(f"Frame {self.frame_count} | "
                          f"FPS: {avg_fps:.1f} | "
                          f"Active: {len(self.trackers)} | "
                          f"Total: {total}")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save debug frame
                    filename = f"{self.debug_folder}/debug_{self.frame_count}_{int(time.time())}.jpg"
                    cv2.imwrite(filename, vis_frame)
                    print(f"Saved debug frame: {filename}")
                
            else:
                # No frame available
                time.sleep(0.001)
        
        cv2.destroyAllWindows()
    
    def start(self):
        """Start the doorway counter"""
        if not self.connect_to_pi():
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
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        cv2.destroyAllWindows()
        
        # Print summary
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
                
                # Average metrics from log
                if self.crossing_log:
                    avg_confidence = np.mean([c['confidence'] for c in self.crossing_log])
                    avg_duration = np.mean([c['duration'] for c in self.crossing_log])
                    
                    # Count unique people who crossed
                    unique_crossers = len(set(c['track_id'] for c in self.crossing_log))
                    
                    # Get max crossings by single person
                    max_crossings = max((c['total_crossings_by_person'] for c in self.crossing_log), default=0)
                    
                    print(f"\nAverage Metrics:")
                    print(f"  Detection confidence: {avg_confidence:.2f}")
                    print(f"  Crossing duration: {avg_duration:.2f}s")
                    print(f"  Unique people who crossed: {unique_crossers}")
                    print(f"  Max crossings by one person: {max_crossings}")
            
            print("="*60)
            
            # Save crossing log
            if self.crossing_log:
                import json
                log_file = f"{self.debug_folder}/crossing_log_{int(time.time())}.json"
                with open(log_file, 'w') as f:
                    json.dump(self.crossing_log, f, indent=2)
                print(f"\nCrossing log saved to: {log_file}")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Overhead Doorway Person Counter with YOLO')
    parser.add_argument('--ip', type=str, required=True,
                        help='IP address of Raspberry Pi camera')
    parser.add_argument('--model', type=str, default='combined_yolo11n.pt',
                        help='Path to YOLO model file (default: combined_yolo11n.pt)')
    parser.add_argument('--port', type=int, default=8485,
                        help='Port number (default: 8485)')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    parser.add_argument('--conf', type=float, default=0.3,
                        help='YOLO confidence threshold (default: 0.5)')
    parser.add_argument('--iou', type=float, default=0.3,
                        help='YOLO IOU threshold for NMS (default: 0.45)')
    
    # API configuration arguments
    parser.add_argument('--api-url', type=str, default='http://localhost:5000',
                        help='Base URL for the counting API (default: http://localhost:5000)')
    parser.add_argument('--api-user', type=str, default='Demo',
                        help='API username (default: Demo)')
    parser.add_argument('--api-password', type=str, default='password',
                        help='API password (default: password)')
    parser.add_argument('--no-api', action='store_true',
                        help='Disable API calls')
    
    args = parser.parse_args()
    
    # Create and start counter
    counter = OverheadDoorwayCounter(args.ip, args.model, args.port)
    counter.debug_mode = args.debug
    counter.doorway_config['yolo_conf_threshold'] = args.conf
    counter.doorway_config['yolo_iou_threshold'] = args.iou
    
    # Configure API settings
    counter.api_config['base_url'] = args.api_url
    counter.api_config['user_id'] = args.api_user
    counter.api_config['password'] = args.api_password
    counter.api_config['enabled'] = not args.no_api
    
    if counter.api_config['enabled']:
        print(f"API Enabled: {counter.api_config['base_url']}")
        print(f"API User: {counter.api_config['user_id']}")
    else:
        print("API Calls Disabled")
    
    try:
        counter.start()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        counter.cleanup()

if __name__ == "__main__":
    # If running directly without arguments, use default IP
    import sys
    if len(sys.argv) == 1:
        # Default for testing
        pi_ip = '174.51.165.192'  # Update this
        model_path = 'models/yolo11s.pt'
        
        print(f"Using default IP: {pi_ip}")
        print(f"Using default model: {model_path}")
        print("Run with --help for options")
        
        counter = OverheadDoorwayCounter(pi_ip, model_path)
        try:
            counter.start()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            counter.cleanup()
    else:
        main()