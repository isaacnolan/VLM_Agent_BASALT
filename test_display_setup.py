#!/usr/bin/env python3
"""
Quick test to verify virtual display setup for MineRL.
"""

import os
import sys
import subprocess

def check_display_env():
    """Check if DISPLAY environment variable is set."""
    display = os.environ.get('DISPLAY')
    if display:
        print(f"✓ DISPLAY is set to: {display}")
        return True
    else:
        print("✗ DISPLAY is not set")
        print("  Run: export DISPLAY=:99")
        return False

def check_xvfb_running():
    """Check if Xvfb is running."""
    try:
        result = subprocess.run(['pgrep', '-f', 'Xvfb'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            print(f"✓ Xvfb is running (PIDs: {', '.join(pids)})")
            return True
        else:
            print("✗ Xvfb is not running")
            print("  Run: source setup_display.sh")
            return False
    except Exception as e:
        print(f"✗ Error checking Xvfb: {e}")
        return False

def check_xvfb_installed():
    """Check if Xvfb is installed."""
    try:
        result = subprocess.run(['which', 'Xvfb'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            path = result.stdout.strip()
            print(f"✓ Xvfb is installed: {path}")
            return True
        else:
            print("✗ Xvfb is not installed")
            print("  CentOS/RHEL: sudo yum install xorg-x11-server-Xvfb mesa-dri-drivers")
            print("  Ubuntu/Debian: sudo apt-get install xvfb mesa-utils")
            return False
    except Exception as e:
        print(f"✗ Error checking Xvfb installation: {e}")
        return False

def check_display_connection():
    """Try to connect to the display."""
    display = os.environ.get('DISPLAY')
    if not display:
        print("✗ Cannot test display connection - DISPLAY not set")
        return False
    
    try:
        result = subprocess.run(['xdpyinfo', '-display', display],
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ Display {display} is accessible")
            # Extract screen info
            for line in result.stdout.split('\n'):
                if 'dimensions:' in line:
                    print(f"  {line.strip()}")
                    break
            return True
        else:
            print(f"✗ Cannot connect to display {display}")
            print(f"  Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ Connection to display {display} timed out")
        return False
    except FileNotFoundError:
        print("⚠ xdpyinfo not found (optional)")
        return True  # Don't fail on this
    except Exception as e:
        print(f"✗ Error testing display: {e}")
        return False

def main():
    """Run all checks."""
    print("="*60)
    print("MineRL Display Setup Verification")
    print("="*60)
    print()
    
    checks = [
        ("Xvfb Installation", check_xvfb_installed),
        ("DISPLAY Environment Variable", check_display_env),
        ("Xvfb Process", check_xvfb_running),
        ("Display Connection", check_display_connection),
    ]
    
    results = []
    for name, check_func in checks:
        print(f"\n{name}:")
        print("-" * 40)
        result = check_func()
        results.append(result)
    
    print()
    print("="*60)
    
    if all(results):
        print("✓ All checks passed! Display is ready for MineRL.")
        print()
        print("You can now run:")
        print("  python qwen_policy_client.py")
        print("  python test_QwenVLM.py --task FindCave")
        return 0
    else:
        print("✗ Some checks failed. Please fix the issues above.")
        print()
        print("Quick fix:")
        print("  1. Install Xvfb if missing")
        print("  2. Run: source setup_display.sh")
        print("  3. Re-run this test: python test_display_setup.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
