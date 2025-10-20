# CrowdCount Server Side (Receiver & Tracker)

This directory contains the receiver and tracking system for CrowdCount that runs on your computer (laptop/desktop with GPU recommended).

## Quick Setup

### Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure you have a YOLO model:**
   - Place your YOLO model file (e.g., `combined_yolo11n.pt` or `yolo11s.pt`) in the project directory
   - Or it will download automatically on first run

3. **Run the receiver:**
   ```bash
   python auto_tracker_receiver.py
   ```

That's it! The receiver will advertise itself on the network and wait for camera connections.

## Files

### Receiver Scripts

- **`auto_tracker_receiver.py`** - Auto-discovery receiver (recommended)
- **`tracker_5_receiver.py`** - Manual IP receiver

### Requirements

- **`requirements.txt`** - Python dependencies

## Usage

### Auto-Discovery Mode (Recommended)

No configuration needed! Just run:

```bash
python auto_tracker_receiver.py
```

The receiver will:
1. Load the YOLO model
2. Advertise itself on the network
3. Wait for camera client to connect
4. Start processing and counting

Options:
```bash
python auto_tracker_receiver.py \
    --model combined_yolo11n.pt \
    --port 8485 \
    --conf 0.3 \
    --iou 0.45
```

### Manual IP Mode

If auto-discovery doesn't work:

```bash
python tracker_5_receiver.py --ip <YOUR_COMPUTER_IP>
```

## Configuration

### Doorway Detection Parameters

You can adjust these in the code or via command-line:

- **`--conf`** - YOLO confidence threshold (default: 0.3)
- **`--iou`** - YOLO IOU threshold for NMS (default: 0.45)
- **`--port`** - Network port (default: 8485)

### Tracking Parameters

In the code, you can adjust `doorway_config`:

```python
self.doorway_config = {
    'doorway_center': 0.5,      # Center line position (0-1)
    'doorway_width': 0.02,      # Width of doorway line
    'approach_zone': 0.15,      # Approach detection zone
    'min_path_length': 50,      # Minimum pixels traveled
    'min_displacement': 100,    # Minimum horizontal movement
    'max_tracking_distance': 120,  # Max tracking distance
    'tracker_timeout': 2.0,     # Seconds before removing tracker
}
```

## Controls

While running:

- **`q`** - Quit the application
- **`s`** - Save current frame as debug image

## Output

### Real-time Display

- Live video with tracking visualization
- Zone indicators (left, doorway, right)
- Tracker IDs and trajectories
- Crossing counts and statistics
- FPS counter

### Console Output

- Crossing events with details
- Direction (left-to-right or right-to-left)
- Confidence scores
- Path metrics
- Running totals

### Saved Data

- **Debug frames** - Saved in `doorway_debug/` folder
- **Crossing log** - JSON file with all crossing events

## System Requirements

### Minimum

- Python 3.8+
- 8GB RAM
- Modern CPU

### Recommended

- Python 3.9+
- 16GB RAM
- NVIDIA GPU with CUDA support
- SSD storage

### GPU Acceleration

For better performance, install PyTorch with CUDA:

```bash
# Check your CUDA version first
nvidia-smi

# Install PyTorch with CUDA (adjust version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Network Requirements

- Same WiFi network as the camera
- mDNS/Bonjour enabled
- Port 8485 open (or configured port)

## Troubleshooting

### No camera detected?

1. **Ensure camera is running first:**
   ```bash
   # On the Pi, run:
   python3 auto_stream.py
   ```

2. **Check both devices are on same network**

3. **Check firewall allows port 8485**

### YOLO model not found?

1. **Download a model:**
   ```bash
   # Using yolo CLI
   yolo detect predict model=yolo11n.pt
   ```

2. **Or specify path:**
   ```bash
   python auto_tracker_receiver.py --model /path/to/your/model.pt
   ```

### Low FPS / Performance issues?

1. **Use GPU acceleration** (install CUDA PyTorch)
2. **Use smaller YOLO model:**
   - `yolo11n.pt` (nano - fastest)
   - `yolo11s.pt` (small)
   - `yolo11m.pt` (medium)
   - `yolo11l.pt` (large)
   - `yolo11x.pt` (extra large - best accuracy)

3. **Adjust confidence threshold:**
   ```bash
   python auto_tracker_receiver.py --conf 0.5
   ```

### Incorrect counting?

1. **Adjust doorway parameters** in the code
2. **Increase min_path_length** for longer tracking
3. **Adjust confidence threshold** for better detections
4. **Use better YOLO model** (yolo11m.pt or larger)

## Model Selection

| Model | Speed | Accuracy | Recommended For |
|-------|-------|----------|----------------|
| yolo11n.pt | Fastest | Good | CPU, Testing |
| yolo11s.pt | Fast | Better | Laptop GPU |
| yolo11m.pt | Medium | Great | Desktop GPU |
| yolo11l.pt | Slow | Excellent | High-end GPU |
| yolo11x.pt | Slowest | Best | Research |

## Advanced Usage

### Custom Training

If you've trained a custom YOLO model for head detection:

```bash
python auto_tracker_receiver.py --model my_custom_model.pt
```

### Multiple Cameras

Run multiple receivers on different ports:

```bash
# Terminal 1
python auto_tracker_receiver.py --port 8485

# Terminal 2
python auto_tracker_receiver.py --port 8486
```

### Data Analysis

The crossing log JSON file can be analyzed:

```python
import json
import pandas as pd

with open('doorway_debug/crossing_log_*.json') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(df.groupby('direction').size())
```

## Support

For issues:
- Check the troubleshooting section
- Run with `--help` for options
- Review debug frames in `doorway_debug/` folder
- Check crossing log for event details
