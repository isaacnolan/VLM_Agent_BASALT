#!/bin/bash
# Setup virtual display for MineRL on headless servers

echo "Setting up virtual display for MineRL..."

# Check if Xvfb is installed
if ! command -v Xvfb &> /dev/null; then
    echo "ERROR: Xvfb is not installed."
    echo "Please install it first:"
    echo "  CentOS/RHEL: sudo yum install xorg-x11-server-Xvfb mesa-dri-drivers"
    echo "  Ubuntu/Debian: sudo apt-get install xvfb mesa-utils"
    exit 1
fi

# Kill any existing Xvfb on display :99
pkill -f "Xvfb :99" 2>/dev/null || true
sleep 1

# Start Xvfb on display :99 with OpenGL support
echo "Starting Xvfb virtual display..."
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Wait for Xvfb to start
sleep 2

# Check if Xvfb is running
if ps -p $XVFB_PID > /dev/null; then
    echo "✓ Xvfb started successfully (PID: $XVFB_PID)"
    echo "✓ Display :99 is ready"
    echo ""
    echo "To use this display, run:"
    echo "  export DISPLAY=:99"
    echo ""
    echo "Or source this script with:"
    echo "  source setup_display.sh"
    echo ""
    echo "To stop Xvfb later:"
    echo "  kill $XVFB_PID"
    
    # Export for current shell if sourced
    export DISPLAY=:99
    echo ""
    echo "DISPLAY is now set to :99 in this session"
else
    echo "✗ Failed to start Xvfb"
    exit 1
fi

