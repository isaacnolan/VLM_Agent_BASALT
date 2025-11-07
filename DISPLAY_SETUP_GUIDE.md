# MineRL Display Setup Guide

## The Problem

MineRL/Minecraft requires a graphical display (X11) to run, but your server is headless (no physical display). This causes the crash:

```
Backend API: NO CONTEXT
GL Caps: 
...
Game crashed!
```

## The Solution

Use **Xvfb** (X Virtual Frame Buffer) to create a virtual display.

## Step-by-Step Setup

### 1. Check if Xvfb is Installed

```bash
which Xvfb
```

If not found, install it:

**CentOS/RHEL:**
```bash
sudo yum install xorg-x11-server-Xvfb mesa-dri-drivers
```

**Ubuntu/Debian:**
```bash
sudo apt-get install xvfb mesa-utils
```

### 2. Option A: Manual Setup (Each Terminal Session)

Every time you open a new terminal to run MineRL:

```bash
# Start Xvfb in background
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &

# Export DISPLAY variable
export DISPLAY=:99

# Now run your MineRL scripts
python qwen_policy_client.py
```

### 3. Option B: Use the Setup Script (Recommended)

**Source the script** to set DISPLAY in your current shell:

```bash
source setup_display.sh
```

This will:
- Start Xvfb on display :99
- Export DISPLAY=:99 in your current shell
- Keep Xvfb running in the background

Then run your MineRL scripts normally:
```bash
python qwen_policy_client.py
python test_QwenVLM.py --task FindCave
```

### 4. Option C: Use the Python Wrapper (Easiest)

The Python wrapper automatically sets up the display:

```bash
python run_with_display.py python qwen_policy_client.py
python run_with_display.py python test_QwenVLM.py --task FindCave
```

This will:
- Check if Xvfb is installed
- Start Xvfb if DISPLAY is not set
- Run your command with the virtual display
- Clean up Xvfb when done

## Testing the Display

After setting up the display, verify it's working:

```bash
# Check DISPLAY is set
echo $DISPLAY
# Should output: :99

# Check Xvfb is running
ps aux | grep Xvfb
# Should show Xvfb process

# Test with a simple X application
xdpyinfo -display :99
# Should show display info without errors
```

## Common Issues

### Issue 1: "DISPLAY not set"

**Solution:** Export DISPLAY before running MineRL
```bash
export DISPLAY=:99
```

### Issue 2: "Xvfb already running"

**Solution:** Kill existing Xvfb and restart
```bash
pkill -f "Xvfb :99"
source setup_display.sh
```

### Issue 3: "Cannot open display :99"

**Solution:** Xvfb might not have started properly
```bash
# Kill any existing Xvfb
pkill Xvfb

# Start with verbose output to see errors
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset

# In another terminal, export DISPLAY
export DISPLAY=:99
```

### Issue 4: "GLX extension missing"

**Solution:** Install Mesa drivers
```bash
# CentOS/RHEL
sudo yum install mesa-dri-drivers mesa-libGL

# Ubuntu/Debian
sudo apt-get install mesa-utils libgl1-mesa-glx
```

## Permanent Setup (Optional)

To automatically set DISPLAY in all your shells, add to `~/.bashrc`:

```bash
# Add to ~/.bashrc
export DISPLAY=:99

# Start Xvfb if not running
if ! pgrep -x "Xvfb" > /dev/null; then
    Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &
fi
```

Then reload:
```bash
source ~/.bashrc
```

## Verification Checklist

Before running MineRL tests:

- [ ] Xvfb is installed: `which Xvfb`
- [ ] Xvfb is running: `ps aux | grep Xvfb`
- [ ] DISPLAY is set: `echo $DISPLAY` shows `:99`
- [ ] Display works: `xdpyinfo -display :99` succeeds

## Quick Reference

### Start Everything
```bash
source setup_display.sh
```

### Run MineRL Scripts
```bash
# Using wrapper (automatic)
python run_with_display.py python qwen_policy_client.py

# Manual (after setup_display.sh)
python qwen_policy_client.py
python test_QwenVLM.py --task FindCave
```

### Stop Everything
```bash
pkill Xvfb
```

## Troubleshooting Commands

```bash
# Check what's using display :99
lsof -i :6099  # X11 typically uses port 6000 + display number

# Check Xvfb logs
Xvfb :99 -screen 0 1024x768x24 -ac +extension GLX +render -noreset 2>&1 | tee xvfb.log

# Test OpenGL support
DISPLAY=:99 glxinfo | grep "OpenGL"

# Run MineRL with error output
DISPLAY=:99 python -u qwen_policy_client.py 2>&1 | tee minerl.log
```

## Additional Notes

- The virtual display persists until you kill Xvfb or logout
- Each terminal session needs `DISPLAY=:99` exported
- Using the Python wrapper is recommended for automated runs
- For batch jobs, use the wrapper or set DISPLAY in your job script

## Example Complete Workflow

```bash
# 1. Setup display (once per session)
source setup_display.sh

# 2. Start QWEN policy server (in one terminal)
python qwen_policy_server.py

# 3. Run client (in another terminal with DISPLAY set)
export DISPLAY=:99
python qwen_policy_client.py

# Or use wrapper (no need to export DISPLAY)
python run_with_display.py python qwen_policy_client.py
```
