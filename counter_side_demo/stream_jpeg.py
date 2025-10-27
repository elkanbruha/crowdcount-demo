#!/usr/bin/env python3
"""
Raspberry Pi 5 Camera Streamer - Updated for compatibility
Run this script on your Raspberry Pi 5 to capture and stream camera footage
This version sends both frame and timestamp for proper tracking
"""

import cv2
import socket
import pickle
import struct
import threading
import time

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False
    print("picamera2 not available, will use legacy camera")

class CameraStreamer:
    def __init__(self, host='0.0.0.0', port=8485, quality=80):
        self.host = host
        self.port = port
        self.quality = quality
        self.server_socket = None
        self.client_socket = None
        self.picam2 = None
        self.cap = None
        self.running = False
        self.use_legacy_camera = False
        
    def check_libcamera_cameras(self):
        """Check if any cameras are available with libcamera"""
        if not PICAMERA2_AVAILABLE:
            return False
        
        try:
            # Try to create a Picamera2 instance to check for cameras
            test_cam = Picamera2()
            camera_info = test_cam.global_camera_info()
            test_cam.close()
            return len(camera_info) > 0
        except Exception:
            return False
    
    def setup_camera(self):
        """Initialize the Pi Camera - try libcamera first, then legacy"""
        # First try libcamera (picamera2)
        if self.check_libcamera_cameras():
            try:
                self.picam2 = Picamera2()
                # Configure camera for video capture
                config = self.picam2.create_video_configuration(
                    main={"size": (640, 480), "format": "RGB888"}
                )
                self.picam2.configure(config)
                self.picam2.start()
                self.use_legacy_camera = False
                print("Modern camera (libcamera) initialized successfully")
                return True
            except Exception as e:
                print(f"Error initializing modern camera: {e}")
                if self.picam2:
                    try:
                        self.picam2.close()
                    except:
                        pass
                    self.picam2 = None
        
        # If libcamera failed or no cameras detected, try legacy camera
        print("No libcamera cameras found, trying legacy camera...")
        
        # Try different video device indices (Pi 5 uses higher numbers)
        video_devices = [0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
        
        for device_id in video_devices:
            try:
                print(f"Trying video device {device_id}...")
                self.cap = cv2.VideoCapture(device_id)
                if self.cap.isOpened():
                    # Test if we can actually read a frame
                    ret, test_frame = self.cap.read()
                    if ret and test_frame is not None:
                        # Set resolution
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        self.cap.set(cv2.CAP_PROP_FPS, 30)
                        self.use_legacy_camera = True
                        print(f"Legacy camera initialized successfully on /dev/video{device_id}")
                        return True
                    else:
                        self.cap.release()
                        self.cap = None
                else:
                    if self.cap:
                        self.cap.release()
                        self.cap = None
            except Exception as e:
                if self.cap:
                    self.cap.release()
                    self.cap = None
                continue
        
        print("Error: Could not open any video device")
        return False
    
    def setup_server(self):
        """Setup TCP server socket"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(1)
            print(f"Server listening on {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"Error setting up server: {e}")
            return False
    
    def wait_for_client(self):
        """Wait for client connection"""
        try:
            print("Waiting for client connection...")
            self.client_socket, addr = self.server_socket.accept()
            print(f"Client connected from {addr}")
            return True
        except Exception as e:
            print(f"Error accepting client: {e}")
            return False
    
    def stream_frames(self):
        """Capture and stream frames to client"""
        frame_count = 0
        try:
            while self.running:
                frame = None
                timestamp = time.time()
                
                if self.use_legacy_camera:
                    # Use legacy camera (OpenCV)
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Failed to read frame from legacy camera")
                        continue
                else:
                    # Use modern camera (picamera2)
                    frame = self.picam2.capture_array()
                    # Convert from RGB to BGR for OpenCV
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                if frame is not None:
                    # Encode frame as JPEG for efficiency
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
                    result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
                    
                    if result:
                        # Create frame data dictionary with encoded frame and timestamp
                        frame_data = {
                            'jpeg': encoded_frame,  # Send JPEG encoded frame
                            'timestamp': timestamp
                        }
                        
                        # Serialize frame data
                        data = pickle.dumps(frame_data)
                        
                        # Send frame size first (using ">L" for big-endian)
                        message_size = struct.pack(">L", len(data))
                        
                        try:
                            self.client_socket.sendall(message_size + data)
                            frame_count += 1
                            
                            # Print periodic status
                            if frame_count % 100 == 0:
                                print(f"Sent {frame_count} frames")
                        except socket.error as e:
                            print(f"Socket error while sending: {e}")
                            break
                
                # Small delay to prevent overwhelming the network
                time.sleep(0.033)  # ~30 FPS
                
        except Exception as e:
            print(f"Error streaming frames: {e}")
        finally:
            print(f"Total frames sent: {frame_count}")
            self.cleanup()
    
    def start_streaming(self):
        """Start the camera streaming process"""
        if not self.setup_camera():
            return False
            
        if not self.setup_server():
            return False
            
        if not self.wait_for_client():
            return False
            
        self.running = True
        print("Starting video stream...")
        
        # Start streaming in a separate thread
        stream_thread = threading.Thread(target=self.stream_frames)
        stream_thread.daemon = True
        stream_thread.start()
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            self.running = False
            
        return True
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        
        if self.picam2:
            try:
                self.picam2.stop()
                self.picam2.close()
            except:
                pass
                
        if self.cap:
            try:
                self.cap.release()
            except:
                pass
            
        if self.client_socket:
            self.client_socket.close()
            
        if self.server_socket:
            self.server_socket.close()
            
        print("Cleanup completed")
        
def main():
    # Get Pi's IP address
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print(f"Pi IP address: {local_ip}")
    print("Make sure your Mac connects to this IP address")
    
    # Create and start streamer
    streamer = CameraStreamer(host='0.0.0.0', port=8485, quality=80)
    
    try:
        streamer.start_streaming()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        streamer.cleanup()

if __name__ == "__main__":
    main()