#!/bin/bash
################################################################################
# CrowdCount Raspberry Pi Setup Script
# This script sets up your Raspberry Pi for CrowdCount camera streaming
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print colored output
print_header() {
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

################################################################################
# Main Setup
################################################################################

print_header "CrowdCount Raspberry Pi Setup"

echo "This script will:"
echo "  1. Check system requirements"
echo "  2. Update system packages (optional)"
echo "  3. Install Python dependencies"
echo "  4. Configure camera (if needed)"
echo "  5. Test camera functionality"
echo "  6. Create auto-start service (optional)"
echo ""

read -p "Continue with setup? [y/N] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 0
fi

################################################################################
# 1. Check System Requirements
################################################################################

print_header "Checking System Requirements"

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    print_success "Python 3 found: $PYTHON_VERSION"
else
    print_error "Python 3 not found. Please install Python 3.7 or higher."
    exit 1
fi

# Check pip
if command -v pip3 &> /dev/null; then
    print_success "pip3 found"
else
    print_warning "pip3 not found. Installing pip..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

# Check if running on Raspberry Pi
if [[ -f /proc/device-tree/model ]] && grep -q "Raspberry Pi" /proc/device-tree/model; then
    PI_MODEL=$(cat /proc/device-tree/model)
    print_success "Running on: $PI_MODEL"
else
    print_warning "Not running on Raspberry Pi (or unable to detect). Continuing anyway..."
fi

################################################################################
# 2. Update System Packages (Optional)
################################################################################

print_header "System Package Updates"

read -p "Update system packages? (Recommended but takes time) [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Updating system packages..."
    sudo apt-get update
    sudo apt-get upgrade -y
    print_success "System packages updated"
else
    print_info "Skipping system update"
fi

################################################################################
# 3. Install System Dependencies
################################################################################

print_header "Installing System Dependencies"

print_info "Installing required system packages..."

# Install OpenCV dependencies
sudo apt-get install -y \
    libopencv-dev \
    python3-opencv \
    libatlas-base-dev \
    libjasper-dev \
    libqt4-test \
    libqtgui4 \
    python3-pyqt5 \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev

# Install network discovery dependencies
sudo apt-get install -y \
    avahi-daemon \
    avahi-utils

print_success "System dependencies installed"

################################################################################
# 4. Install Python Dependencies
################################################################################

print_header "Installing Python Dependencies"

print_info "Installing Python packages from requirements.txt..."

# Upgrade pip
python3 -m pip install --upgrade pip

# Install from requirements.txt
if [[ -f "$SCRIPT_DIR/requirements.txt" ]]; then
    # For Pi, prefer headless OpenCV for better performance
    print_info "Installing opencv-python-headless (better for headless Pi)..."
    pip3 install opencv-python-headless>=4.8.0 --break-system-packages

    # Install other requirements
    pip3 install zeroconf>=0.131.0 numpy>=1.24.0 --break-system-packages

    print_success "Python dependencies installed"
else
    print_error "requirements.txt not found in $SCRIPT_DIR"
    exit 1
fi

################################################################################
# 5. Camera Configuration
################################################################################

print_header "Camera Configuration"

# Check if camera is enabled (Pi 5 and older models)
print_info "Checking camera configuration..."

# For newer Raspberry Pi OS (Bullseye and later)
if command -v libcamera-hello &> /dev/null; then
    print_success "libcamera detected (modern Pi camera system)"
    print_info "Testing camera with libcamera..."

    if timeout 3s libcamera-hello --list-cameras 2>&1 | grep -q "Available cameras"; then
        print_success "Camera detected and working!"
    else
        print_warning "Camera not detected. Please ensure camera is connected and enabled."
        print_info "Run 'sudo raspi-config' -> Interface Options -> Camera to enable"
    fi
else
    # Legacy camera system
    print_info "Checking legacy camera configuration..."

    if grep -q "^start_x=1" /boot/config.txt 2>/dev/null || \
       grep -q "^camera_auto_detect=1" /boot/config.txt 2>/dev/null; then
        print_success "Camera appears to be enabled in config"
    else
        print_warning "Camera might not be enabled."
        print_info "Run 'sudo raspi-config' -> Interface Options -> Legacy Camera to enable"
    fi
fi

################################################################################
# 6. Test Camera
################################################################################

print_header "Testing Camera"

read -p "Test camera capture? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Creating test script..."

    cat > /tmp/test_camera.py << 'EOF'
import cv2
import sys

print("Attempting to open camera 0...")
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("✗ Failed to open camera 0")
    sys.exit(1)

print("✓ Camera opened successfully")

# Try to read a frame
ret, frame = camera.read()

if ret and frame is not None:
    print(f"✓ Successfully captured frame: {frame.shape}")
    print("✓ Camera is working!")
    camera.release()
    sys.exit(0)
else:
    print("✗ Failed to capture frame")
    camera.release()
    sys.exit(1)
EOF

    python3 /tmp/test_camera.py

    if [ $? -eq 0 ]; then
        print_success "Camera test passed!"
    else
        print_error "Camera test failed. Please check camera connection and configuration."
    fi

    rm /tmp/test_camera.py
else
    print_info "Skipping camera test"
fi

################################################################################
# 7. Create Auto-Start Service (Optional)
################################################################################

print_header "Auto-Start Service"

read -p "Create systemd service for auto-start on boot? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Creating systemd service..."

    SERVICE_FILE="/etc/systemd/system/crowdcount-camera.service"

    sudo tee $SERVICE_FILE > /dev/null << EOF
[Unit]
Description=CrowdCount Camera Streamer
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$SCRIPT_DIR
ExecStart=/usr/bin/python3 $SCRIPT_DIR/auto_stream.py
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable crowdcount-camera.service

    print_success "Service created and enabled"
    print_info "Control the service with:"
    print_info "  sudo systemctl start crowdcount-camera    # Start now"
    print_info "  sudo systemctl stop crowdcount-camera     # Stop service"
    print_info "  sudo systemctl status crowdcount-camera   # Check status"
    print_info "  sudo journalctl -u crowdcount-camera -f   # View logs"
else
    print_info "Skipping service creation"
fi

################################################################################
# 8. Final Instructions
################################################################################

print_header "Setup Complete!"

echo -e "${GREEN}✓ CrowdCount Pi setup completed successfully!${NC}\n"

print_info "Next steps:"
echo ""
echo "1. Make sure the receiver is running on your computer:"
echo "   ${YELLOW}python auto_tracker_receiver.py${NC}"
echo ""
echo "2. Start the camera streamer on this Pi:"
echo "   ${YELLOW}cd $SCRIPT_DIR${NC}"
echo "   ${YELLOW}python3 auto_stream.py${NC}"
echo ""
echo "3. Both devices should auto-discover each other on the network!"
echo ""

if systemctl is-enabled crowdcount-camera.service &> /dev/null; then
    echo "4. Or start the service (will auto-start on boot):"
    echo "   ${YELLOW}sudo systemctl start crowdcount-camera${NC}"
    echo ""
fi

print_info "Troubleshooting:"
echo "  - Ensure both devices are on the same WiFi network"
echo "  - Check camera with: v4l2-ctl --list-devices"
echo "  - Test manually: python3 auto_stream.py"
echo "  - View logs: sudo journalctl -u crowdcount-camera -f"
echo ""

print_success "Happy crowd counting! 📹"
