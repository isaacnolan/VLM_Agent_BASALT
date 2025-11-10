"""
Test script for the history-enabled policy server API.

This demonstrates how to send requests with state-action history.
"""

import requests
import base64
import json
from PIL import Image
import io
import numpy as np


def encode_image(image_path_or_array):
    """Encode an image to base64 string."""
    if isinstance(image_path_or_array, str):
        # Load from file
        with open(image_path_or_array, 'rb') as f:
            image_bytes = f.read()
    elif isinstance(image_path_or_array, np.ndarray):
        # Convert numpy array to image
        img = Image.fromarray(image_path_or_array.astype(np.uint8))
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
    else:
        raise ValueError("Image must be file path or numpy array")
    
    return base64.b64encode(image_bytes).decode('utf-8')


def test_single_image_mode():
    """Test backward compatibility with single image."""
    print("Testing single image mode (backward compatibility)...")
    
    # Create a dummy image
    dummy_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    encoded_image = encode_image(dummy_image)
    
    request_data = {
        "task_name": "MineRLBasaltFindCave-v0",
        "image": {
            "data": encoded_image
        },
        "step": 1,
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    response = requests.post(
        "http://localhost:8001/get_action",
        json=request_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Single image mode works!")
        print(f"  Action: {result['action']}")
        print(f"  Reasoning: {result['reasoning'][:100]}...")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")


def test_history_mode():
    """Test new history mode with state-action pairs."""
    print("\nTesting history mode with state-action pairs...")
    
    # Create a history of 3 state-action pairs
    history = []
    
    # Frame 1: Initial observation, no action yet
    image1 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    history.append({
        "image": {"data": encode_image(image1)},
        "action": None  # Initial state
    })
    
    # Frame 2: After moving forward
    image2 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    history.append({
        "image": {"data": encode_image(image2)},
        "action": {
            "forward": 1,
            "back": 0,
            "left": 0,
            "right": 0,
            "jump": 0,
            "sneak": 0,
            "sprint": 0,
            "attack": 0,
            "use": 0,
            "drop": 0,
            "inventory": 0,
            "camera": [0.0, 0.0],
            "hotbar.1": 0,
            "hotbar.2": 0,
            "hotbar.3": 0,
            "hotbar.4": 0,
            "hotbar.5": 0,
            "hotbar.6": 0,
            "hotbar.7": 0,
            "hotbar.8": 0,
            "hotbar.9": 0,
            "ESC": 0
        }
    })
    
    # Frame 3: After turning camera
    image3 = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    history.append({
        "image": {"data": encode_image(image3)},
        "action": {
            "forward": 0,
            "back": 0,
            "left": 0,
            "right": 0,
            "jump": 0,
            "sneak": 0,
            "sprint": 0,
            "attack": 0,
            "use": 0,
            "drop": 0,
            "inventory": 0,
            "camera": [5.0, -2.0],  # Turned right and looked up
            "hotbar.1": 0,
            "hotbar.2": 0,
            "hotbar.3": 0,
            "hotbar.4": 0,
            "hotbar.5": 0,
            "hotbar.6": 0,
            "hotbar.7": 0,
            "hotbar.8": 0,
            "hotbar.9": 0,
            "ESC": 0
        }
    })
    
    request_data = {
        "task_name": "MineRLBasaltFindCave-v0",
        "history": history,
        "step": 3,
        "temperature": 0.7,
        "max_tokens": 512,
        "max_history_length": 5
    }
    
    response = requests.post(
        "http://localhost:8001/get_action",
        json=request_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ History mode works!")
        print(f"  Processed {len(history)} historical frames")
        print(f"  Action: {result['action']}")
        print(f"  Reasoning: {result['reasoning'][:100]}...")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")


def test_long_history():
    """Test history with more frames than max_history_length."""
    print("\nTesting long history (> max_history_length)...")
    
    # Create a history of 10 frames (exceeds default max of 5)
    history = []
    for i in range(10):
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        history.append({
            "image": {"data": encode_image(image)},
            "action": {
                "forward": 1 if i % 2 == 0 else 0,
                "back": 0,
                "left": 0,
                "right": 0,
                "jump": 0,
                "sneak": 0,
                "sprint": 0,
                "attack": 0,
                "use": 0,
                "drop": 0,
                "inventory": 0,
                "camera": [0.0, 0.0],
                "hotbar.1": 0,
                "hotbar.2": 0,
                "hotbar.3": 0,
                "hotbar.4": 0,
                "hotbar.5": 0,
                "hotbar.6": 0,
                "hotbar.7": 0,
                "hotbar.8": 0,
                "hotbar.9": 0,
                "ESC": 0
            }
        })
    
    request_data = {
        "task_name": "MineRLBasaltFindCave-v0",
        "history": history,
        "step": 10,
        "max_history_length": 3  # Only use last 3 frames
    }
    
    response = requests.post(
        "http://localhost:8001/get_action",
        json=request_data
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ Long history handled correctly!")
        print(f"  Sent {len(history)} frames, limited to 3")
        print(f"  Action: {result['action']}")
        print(f"  Reasoning: {result['reasoning'][:100]}...")
    else:
        print(f"✗ Error: {response.status_code}")
        print(f"  {response.text}")


if __name__ == "__main__":
    print("="*60)
    print("Testing History-Enabled Policy Server API")
    print("="*60)
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8001/health")
        if response.status_code != 200:
            print("Error: Server not healthy")
            exit(1)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server at http://localhost:8001")
        print("Please start the server first: python qwen_policy_server.py")
        exit(1)
    
    print("Server is running!\n")
    
    # Run tests
    test_single_image_mode()
    test_history_mode()
    test_long_history()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
