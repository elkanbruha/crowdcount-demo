#!/usr/bin/env python3
"""
Camera Stream Client for CrowdCount
Streams camera feed to the receiver for person counting
"""

import cv2
import socket
import pickle
import struct
import time
import argparse
import sys

class CameraStreamer:
    """Streams camera feed to a remote receiver"""

    def __init__(self, server_ip: str, port: int = 8485, camera_id: int = 0,
                 resolution: tuple = (640, 480), fps: int = 30, quality: int = 85):
        """
        Initialize camera streamer

        Args:
            server_ip: IP address of the receiver server
            port: Port number for connection
            camera_id: Camera device ID (0 for default camera)
            resolution: Video resolution (width, height)
            fps: Target frames per second
            quality: JPEG quality (1-100, higher is better)
        """
        self.server_ip = server_ip
        self.port = port
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.quality = quality

        self.socket = None
        self.camera = None
        self.running = False

        # Stats
        self.frame_count = 0
        self.start_time = None
        self.last_stats_time = None

    def setup_camera(self) -> bool:
        """Initialize camera"""
        try:
            print(f"Initializing camera {self.camera_id}...")
            self.camera = cv2.VideoCapture(self.camera_id)

            if not self.camera.isOpened():
                print(f" Failed to open camera {self.camera_id}")
                return False

            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)

            # Verify settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.camera.get(cv2.CAP_PROP_FPS))

            print(f" Camera initialized")
            print(f"  Resolution: {actual_width}x{actual_height}")
            print(f"  FPS: {actual_fps}")

            return True

        except Exception as e:
            print(f" Camera initialization failed: {e}")
            return False

    def connect_to_server(self) -> bool:
        """Connect to the receiver server"""
        try:
            print(f"Connecting to server {self.server_ip}:{self.port}...")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)
            self.socket.connect((self.server_ip, self.port))
            self.socket.settimeout(None)

            print(f" Connected to server at {self.server_ip}:{self.port}")
            return True

        except Exception as e:
            print(f" Connection failed: {e}")
            if self.socket:
                self.socket.close()
                self.socket = None
            return False

    def send_frame(self, frame) -> bool:
        """
        Encode and send a frame to the receiver

        Args:
            frame: OpenCV frame (numpy array)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Encode frame as JPEG
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
            result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)

            if not result:
                print(" Failed to encode frame")
                return False

            # Pickle the encoded frame
            data = pickle.dumps(encoded_frame)

            # Send message size first (as packed unsigned long)
            message_size = struct.pack("L", len(data))
            self.socket.sendall(message_size)

            # Send the pickled data
            self.socket.sendall(data)

            return True

        except BrokenPipeError:
            print(" Connection lost (broken pipe)")
            return False
        except Exception as e:
            print(f" Failed to send frame: {e}")
            return False

    def print_stats(self):
        """Print streaming statistics"""
        if self.start_time is None:
            return

        elapsed = time.time() - self.start_time
        if elapsed > 0:
            avg_fps = self.frame_count / elapsed

            # Print stats every 5 seconds
            if self.last_stats_time is None or (time.time() - self.last_stats_time) >= 5.0:
                print(f"Frames: {self.frame_count:,} | "
                      f"FPS: {avg_fps:.1f} | "
                      f"Runtime: {elapsed:.0f}s")
                self.last_stats_time = time.time()

    def stream(self):
        """Main streaming loop"""
        if not self.setup_camera():
            return False

        if not self.connect_to_server():
            self.cleanup()
            return False

        self.running = True
        self.start_time = time.time()
        self.last_stats_time = self.start_time

        print("\n" + "="*60)
        print("CAMERA STREAMING - Started")
        print("="*60)
        print(f"Server: {self.server_ip}:{self.port}")
        print(f"Camera: {self.camera_id}")
        print(f"Resolution: {self.resolution[0]}x{self.resolution[1]}")
        print(f"Quality: {self.quality}")
        print("Press Ctrl+C to stop")
        print("="*60 + "\n")

        try:
            while self.running:
                # Capture frame
                ret, frame = self.camera.read()

                if not ret:
                    print(" Failed to read frame from camera")
                    break

                # Send frame to server
                if not self.send_frame(frame):
                    print(" Failed to send frame, stopping...")
                    break

                self.frame_count += 1

                # Print stats periodically
                self.print_stats()

                # Small delay to control frame rate
                time.sleep(1.0 / self.fps)

        except KeyboardInterrupt:
            print("\n\nStopping stream...")
        except Exception as e:
            print(f"\n Streaming error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

        return True

    def cleanup(self):
        """Clean up resources and print summary"""
        self.running = False

        # Release camera
        if self.camera is not None:
            self.camera.release()
            print(" Camera released")

        # Close socket
        if self.socket is not None:
            try:
                self.socket.close()
                print(" Connection closed")
            except:
                pass

        # Print summary
        if self.start_time is not None:
            elapsed = time.time() - self.start_time

            print("\n" + "="*60)
            print("CAMERA STREAMING - Summary")
            print("="*60)
            print(f"Runtime: {elapsed:.1f} seconds")
            print(f"Frames sent: {self.frame_count:,}")
            if elapsed > 0:
                print(f"Average FPS: {self.frame_count/elapsed:.1f}")
            print("="*60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Stream camera feed to CrowdCount receiver')
    parser.add_argument('--server', type=str, required=True,
                        help='IP address of receiver server')
    parser.add_argument('--port', type=int, default=8485,
                        help='Port number (default: 8485)')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--resolution', type=str, default='640x480',
                        help='Video resolution WIDTHxHEIGHT (default: 640x480)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target frames per second (default: 30)')
    parser.add_argument('--quality', type=int, default=85,
                        help='JPEG quality 1-100 (default: 85)')

    args = parser.parse_args()

    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        print(f" Invalid resolution format: {args.resolution}")
        print("  Use format: WIDTHxHEIGHT (e.g., 640x480)")
        sys.exit(1)

    # Validate quality
    if not 1 <= args.quality <= 100:
        print(f" Quality must be between 1 and 100, got {args.quality}")
        sys.exit(1)

    # Create and start streamer
    streamer = CameraStreamer(
        server_ip=args.server,
        port=args.port,
        camera_id=args.camera,
        resolution=resolution,
        fps=args.fps,
        quality=args.quality
    )

    try:
        streamer.stream()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # If running without arguments, show help
    if len(sys.argv) == 1:
        print("CrowdCount Camera Streamer")
        print("="*60)
        print("This script streams camera feed to a CrowdCount receiver.")
        print("\nUsage:")
        print("  python stream.py --server <IP_ADDRESS>")
        print("\nExample:")
        print("  python stream.py --server 192.168.1.100")
        print("  python stream.py --server 192.168.1.100 --camera 0 --fps 30")
        print("\nFor all options, run:")
        print("  python stream.py --help")
        print("="*60)
        sys.exit(0)

    main()
