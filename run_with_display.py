#!/usr/bin/env python3
"""
Wrapper to run MineRL with virtual display setup.

This script ensures DISPLAY is set and Xvfb is running before launching MineRL tests.
"""

import os
import sys
import subprocess
import time
import signal

def check_xvfb_installed():
    """Check if Xvfb is installed."""
    try:
        subprocess.run(['which', 'Xvfb'], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False

def start_xvfb(display=':99'):
    """Start Xvfb virtual display."""
    print(f"Starting Xvfb on display {display}...")
    
    # Kill any existing Xvfb on this display
    try:
        subprocess.run(['pkill', '-f', f'Xvfb {display}'], stderr=subprocess.DEVNULL)
        time.sleep(1)
    except:
        pass
    
    # Start Xvfb
    xvfb_process = subprocess.Popen([
        'Xvfb', display,
        '-screen', '0', '1024x768x24',
        '-ac',
        '+extension', 'GLX',
        '+render',
        '-noreset'
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Wait for Xvfb to start
    time.sleep(2)
    
    # Check if it's running
    if xvfb_process.poll() is None:
        print(f"✓ Xvfb started successfully (PID: {xvfb_process.pid})")
        return xvfb_process
    else:
        print("✗ Failed to start Xvfb")
        return None

def run_with_display(command):
    """Run a command with virtual display."""
    # Check if DISPLAY is already set
    if 'DISPLAY' not in os.environ:
        print("DISPLAY not set, setting up virtual display...")
        
        # Check Xvfb installation
        if not check_xvfb_installed():
            print("ERROR: Xvfb is not installed!")
            print("Install it with:")
            print("  CentOS/RHEL: sudo yum install xorg-x11-server-Xvfb mesa-dri-drivers")
            print("  Ubuntu/Debian: sudo apt-get install xvfb mesa-utils")
            sys.exit(1)
        
        # Start Xvfb
        xvfb_process = start_xvfb(':99')
        if not xvfb_process:
            sys.exit(1)
        
        # Set DISPLAY environment variable
        os.environ['DISPLAY'] = ':99'
        print("✓ DISPLAY set to :99")
    else:
        print(f"Using existing DISPLAY: {os.environ['DISPLAY']}")
        xvfb_process = None
    
    # Run the command
    print(f"\nRunning command: {' '.join(command)}")
    print("="*60)
    
    try:
        result = subprocess.run(command)
        return_code = result.returncode
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return_code = 130
    finally:
        # Clean up Xvfb if we started it
        if xvfb_process:
            print("\nCleaning up Xvfb...")
            xvfb_process.terminate()
            try:
                xvfb_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                xvfb_process.kill()
            print("✓ Xvfb stopped")
    
    return return_code

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_with_display.py <command> [args...]")
        print("\nExamples:")
        print("  python run_with_display.py python qwen_policy_client.py")
        print("  python run_with_display.py python test_QwenVLM.py --task FindCave")
        sys.exit(1)
    
    # Get command from arguments
    command = sys.argv[1:]
    
    # Run with display
    exit_code = run_with_display(command)
    sys.exit(exit_code)
