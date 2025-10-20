#!/usr/bin/env python3
"""
Auto-Discovery Camera Stream Client for CrowdCount
Automatically discovers and connects to CrowdCount receiver on the network
"""

import cv2
import socket
import pickle
import struct
import time
import argparse
import sys
from zeroconf import Zeroconf, ServiceBrowser, ServiceListener
import threading

class CrowdCountServiceListener(ServiceListener):
    """Listener for CrowdCount receiver services on the network"""

    def __init__(self):
        self.services = {}
        self.service_found = threading.Event()
        self.lock = threading.Lock()

    def add_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        """Called when a service is discovered"""
        info = zc.get_service_info(service_type, name)
        if info:
            with self.lock:
                # Parse the IP address
                if info.addresses:
                    ip_address = socket.inet_ntoa(info.addresses[0])
                    port = info.port

                    # Get service properties
                    props = {}
                    if info.properties:
                        props = {k.decode('utf-8'): v.decode('utf-8') for k, v in info.properties.items()}

                    self.services[name] = {
                        'ip': ip_address,
                        'port': port,
                        'name': name,
                        'properties': props
                    }

                    print(f"\n✓ Discovered CrowdCount receiver:")
                    print(f"  Name: {name}")
                    print(f"  IP: {ip_address}")
                    print(f"  Port: {port}")
                    if props:
                        for k, v in props.items():
                            print(f"  {k}: {v}")

                    self.service_found.set()

    def remove_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        """Called when a service is removed"""
        with self.lock:
            if name in self.services:
                print(f"\n✗ Service removed: {name}")
                del self.services[name]

    def update_service(self, zc: Zeroconf, service_type: str, name: str) -> None:
        """Called when a service is updated"""
        pass

    def get_services(self):
        """Get list of discovered services"""
        with self.lock:
            return list(self.services.values())

class AutoCameraStreamer:
    """Auto-discovery camera streamer"""

    def __init__(self, camera_id: int = 0, resolution: tuple = (640, 480),
                 fps: int = 30, quality: int = 85, discovery_timeout: int = 10):
        """
        Initialize auto-discovery camera streamer

        Args:
            camera_id: Camera device ID (0 for default camera)
            resolution: Video resolution (width, height)
            fps: Target frames per second
            quality: JPEG quality (1-100)
            discovery_timeout: Timeout for service discovery in seconds
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.quality = quality
        self.discovery_timeout = discovery_timeout

        self.socket = None
        self.camera = None
        self.running = False

        # Discovery
        self.zeroconf = None
        self.browser = None
        self.listener = None

        # Stats
        self.frame_count = 0
        self.start_time = None
        self.last_stats_time = None

    def discover_receiver(self) -> dict:
        """
        Discover CrowdCount receiver on the network

        Returns:
            Service info dict or None if not found
        """
        print("="*60)
        print("Searching for CrowdCount receiver on network...")
        print(f"Timeout: {self.discovery_timeout} seconds")
        print("="*60 + "\n")

        try:
            self.zeroconf = Zeroconf()
            self.listener = CrowdCountServiceListener()
            self.browser = ServiceBrowser(self.zeroconf, "_crowdcount._tcp.local.", self.listener)

            # Wait for service to be found
            found = self.listener.service_found.wait(timeout=self.discovery_timeout)

            if found:
                services = self.listener.get_services()
                if services:
                    # Return first service found
                    return services[0]

            print(f"\n✗ No CrowdCount receiver found within {self.discovery_timeout} seconds")
            return None

        except Exception as e:
            print(f"✗ Discovery error: {e}")
            return None

    def cleanup_discovery(self):
        """Clean up discovery resources"""
        if self.browser:
            self.browser.cancel()

        if self.zeroconf:
            self.zeroconf.close()

    def setup_camera(self) -> bool:
        """Initialize camera"""
        try:
            print(f"\nInitializing camera {self.camera_id}...")
            self.camera = cv2.VideoCapture(self.camera_id)

            if not self.camera.isOpened():
                print(f"✗ Failed to open camera {self.camera_id}")
                return False

            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, self.fps)

            # Verify settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.camera.get(cv2.CAP_PROP_FPS))

            print(f"✓ Camera initialized")
            print(f"  Resolution: {actual_width}x{actual_height}")
            print(f"  FPS: {actual_fps}")

            return True

        except Exception as e:
            print(f"✗ Camera initialization failed: {e}")
            return False

    def connect_to_server(self, server_ip: str, port: int) -> bool:
        """Connect to the receiver server"""
        try:
            print(f"\nConnecting to server {server_ip}:{port}...")
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(10.0)
            self.socket.connect((server_ip, port))
            self.socket.settimeout(None)

            print(f"✓ Connected to server at {server_ip}:{port}")
            return True

        except Exception as e:
            print(f"✗ Connection failed: {e}")
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
                print("✗ Failed to encode frame")
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
            print("✗ Connection lost (broken pipe)")
            return False
        except Exception as e:
            print(f"✗ Failed to send frame: {e}")
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
        """Main streaming loop with auto-discovery"""
        # Discover receiver
        service = self.discover_receiver()

        if not service:
            print("\n" + "="*60)
            print("TROUBLESHOOTING:")
            print("="*60)
            print("1. Make sure the receiver is running on another machine")
            print("2. Ensure both devices are on the same network")
            print("3. Check firewall settings allow mDNS/Bonjour")
            print("4. Try increasing --discovery-timeout")
            print("="*60)
            self.cleanup_discovery()
            return False

        server_ip = service['ip']
        server_port = service['port']

        # Clean up discovery (we don't need it anymore)
        self.cleanup_discovery()

        # Setup camera
        if not self.setup_camera():
            return False

        # Connect to server
        if not self.connect_to_server(server_ip, server_port):
            self.cleanup()
            return False

        self.running = True
        self.start_time = time.time()
        self.last_stats_time = self.start_time

        print("\n" + "="*60)
        print("CAMERA STREAMING - Started (Auto-Discovery)")
        print("="*60)
        print(f"Server: {server_ip}:{server_port}")
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
                    print("✗ Failed to read frame from camera")
                    break

                # Send frame to server
                if not self.send_frame(frame):
                    print("✗ Failed to send frame, stopping...")
                    break

                self.frame_count += 1

                # Print stats periodically
                self.print_stats()

                # Small delay to control frame rate
                time.sleep(1.0 / self.fps)

        except KeyboardInterrupt:
            print("\n\nStopping stream...")
        except Exception as e:
            print(f"\n✗ Streaming error: {e}")
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
            print("✓ Camera released")

        # Close socket
        if self.socket is not None:
            try:
                self.socket.close()
                print("✓ Connection closed")
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
    parser = argparse.ArgumentParser(description='Auto-Discovery Camera Streamer for CrowdCount')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--resolution', type=str, default='640x480',
                        help='Video resolution WIDTHxHEIGHT (default: 640x480)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Target frames per second (default: 30)')
    parser.add_argument('--quality', type=int, default=85,
                        help='JPEG quality 1-100 (default: 85)')
    parser.add_argument('--discovery-timeout', type=int, default=10,
                        help='Service discovery timeout in seconds (default: 10)')

    args = parser.parse_args()

    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        print(f"✗ Invalid resolution format: {args.resolution}")
        print("  Use format: WIDTHxHEIGHT (e.g., 640x480)")
        sys.exit(1)

    # Validate quality
    if not 1 <= args.quality <= 100:
        print(f"✗ Quality must be between 1 and 100, got {args.quality}")
        sys.exit(1)

    print("="*60)
    print("CrowdCount Auto-Discovery Camera Streamer")
    print("="*60)
    print("This camera will automatically find and connect to the receiver.")
    print("No manual IP configuration needed!")
    print("="*60 + "\n")

    # Create and start streamer
    streamer = AutoCameraStreamer(
        camera_id=args.camera,
        resolution=resolution,
        fps=args.fps,
        quality=args.quality,
        discovery_timeout=args.discovery_timeout
    )

    try:
        success = streamer.stream()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
