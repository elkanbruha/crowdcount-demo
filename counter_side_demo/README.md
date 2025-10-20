# CrowdCount Camera Side (Raspberry Pi)

This directory contains the camera streaming client for CrowdCount that runs on a Raspberry Pi or any device with a camera.

## Quick Setup

### On Raspberry Pi

1. **Transfer files to your Pi:**
   ```bash
   # From your computer, copy the entire counter_side_demo folder to your Pi
   scp -r counter_side_demo/ pi@raspberrypi.local:~/
   ```

2. **SSH into your Pi:**
   ```bash
   ssh pi@raspberrypi.local
   ```

3. **Run the setup script:**
   ```bash
   cd counter_side_demo
   ./setup_pi.sh
   ```

   The setup script will:
   - ✓ Check Python and system requirements
   - ✓ Install all dependencies
   - ✓ Configure the camera
   - ✓ Test camera functionality
   - ✓ Optionally create an auto-start service

4. **Start streaming:**
   ```bash
   python3 auto_stream.py
   ```

That's it! The Pi will automatically discover the receiver on your network.

## Manual Installation

If you prefer to install manually:

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y libopencv-dev python3-opencv avahi-daemon

# Install Python dependencies
pip3 install -r requirements.txt
```

## Files

- **`auto_stream.py`** - Auto-discovery camera streamer (recommended)
- **`stream.py`** - Manual IP camera streamer
- **`setup_pi.sh`** - Automated setup script for Raspberry Pi
- **`requirements.txt`** - Python dependencies

## Usage

### Auto-Discovery Mode (Recommended)

No configuration needed! Just run:

```bash
python3 auto_stream.py
```

Options:
```bash
python3 auto_stream.py --camera 0 --fps 30 --resolution 640x480 --quality 85
```

### Manual IP Mode

If auto-discovery doesn't work:

```bash
python3 stream.py --server <RECEIVER_IP_ADDRESS>
```

## Auto-Start on Boot

The setup script can create a systemd service. To control it:

```bash
# Start the service
sudo systemctl start crowdcount-camera

# Stop the service
sudo systemctl stop crowdcount-camera

# Check status
sudo systemctl status crowdcount-camera

# View logs
sudo journalctl -u crowdcount-camera -f
```

## Troubleshooting

### Camera not working?

1. **Check if camera is detected:**
   ```bash
   v4l2-ctl --list-devices
   ```

2. **Enable camera in raspi-config:**
   ```bash
   sudo raspi-config
   # Navigate to: Interface Options -> Camera
   ```

3. **Test with libcamera (modern Pi OS):**
   ```bash
   libcamera-hello --list-cameras
   libcamera-hello
   ```

### Auto-discovery not working?

1. **Ensure both devices are on the same WiFi network**
2. **Check Avahi daemon is running:**
   ```bash
   sudo systemctl status avahi-daemon
   ```
3. **Increase discovery timeout:**
   ```bash
   python3 auto_stream.py --discovery-timeout 30
   ```
4. **Fall back to manual mode:**
   ```bash
   python3 stream.py --server 192.168.1.100
   ```

### Connection drops or lag?

1. **Lower resolution:**
   ```bash
   python3 auto_stream.py --resolution 320x240
   ```
2. **Lower FPS:**
   ```bash
   python3 auto_stream.py --fps 15
   ```
3. **Lower quality:**
   ```bash
   python3 auto_stream.py --quality 70
   ```

## Network Requirements

- Same WiFi network as the receiver
- mDNS/Bonjour enabled (usually enabled by default)
- Port 8485 accessible (configurable)

## Hardware Requirements

- Raspberry Pi (any model with camera support)
- Camera module or USB webcam
- WiFi connection

## Performance Tips

- Use wired Ethernet for more stable connection
- Lower resolution/FPS if experiencing lag
- Use `opencv-python-headless` for better performance (included in requirements)
- Close other applications running on the Pi

## Support

For issues, check the main CrowdCount documentation or run:
```bash
python3 auto_stream.py --help
```
