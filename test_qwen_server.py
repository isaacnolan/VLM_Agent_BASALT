"""
Simple test script for QWEN Policy Server

This script tests the server without requiring a full MineRL environment.
"""

import requests
import base64
import json
import numpy as np
from PIL import Image
import io

def create_test_image(width=640, height=360):
    """Create a test image that simulates a Minecraft POV."""
    # Create a simple test image with some patterns
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Add some structure (sky, ground, etc.)
    img[0:height//2, :] = [135, 206, 235]  # Sky blue
    img[height//2:, :] = [34, 139, 34]     # Ground green
    
    return Image.fromarray(img)

def encode_image(image):
    """Encode PIL image to base64."""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def test_health_check(server_url="http://localhost:8001"):
    """Test the health check endpoint."""
    print("Testing health check endpoint...")
    try:
        response = requests.get(f"{server_url}/health")
        response.raise_for_status()
        health = response.json()
        print("✓ Health check passed:")
        print(json.dumps(health, indent=2))
        return True
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return False

def test_get_action(server_url="http://localhost:8001", task_name="MineRLBasaltFindCave-v0"):
    """Test the get_action endpoint."""
    print(f"\nTesting get_action endpoint for {task_name}...")
    
    # Create test image
    test_image = create_test_image()
    image_base64 = encode_image(test_image)
    
    # Prepare request
    payload = {
        "task_name": task_name,
        "image": {
            "data": image_base64
        },
        "step": 1,
        "temperature": 0.7,
        "max_tokens": 256
    }
    
    try:
        print("Sending request to server...")
        response = requests.post(
            f"{server_url}/get_action",
            json=payload,
            timeout=60  # Longer timeout for VLM inference
        )
        response.raise_for_status()
        
        result = response.json()
        print("✓ Get action succeeded:")
        print(f"  Reasoning: {result['reasoning']}")
        print(f"  Action keys: {list(result['action'].keys())}")
        print(f"  Sample actions:")
        print(f"    - forward: {result['action']['forward']}")
        print(f"    - camera: {result['action']['camera']}")
        print(f"    - jump: {result['action']['jump']}")
        return True
        
    except requests.exceptions.Timeout:
        print("✗ Request timed out (this is normal for first request - model loading)")
        return False
    except Exception as e:
        print(f"✗ Get action failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"  Response: {e.response.text}")
        return False

def main():
    """Run all tests."""
    server_url = "http://localhost:8001"
    
    print("="*60)
    print("QWEN Policy Server Test Suite")
    print("="*60)
    print(f"Server URL: {server_url}")
    print()
    
    # Test 1: Health check
    health_ok = test_health_check(server_url)
    
    if not health_ok:
        print("\n" + "="*60)
        print("Server is not running or not healthy!")
        print("Please start the server with: python qwen_policy_server.py")
        print("="*60)
        return
    
    # Test 2: Get action
    action_ok = test_get_action(server_url)
    
    print("\n" + "="*60)
    if health_ok and action_ok:
        print("✓ All tests passed!")
        print("The QWEN policy server is working correctly.")
    else:
        print("✗ Some tests failed")
        print("Check the logs above for details.")
    print("="*60)

if __name__ == "__main__":
    main()
