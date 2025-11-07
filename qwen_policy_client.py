"""
Client for QWEN VLM Policy Server

This script demonstrates how to use the QWEN policy server to get actions
for MineRL BASALT tasks.
"""

import requests
import base64
import json
import numpy as np
from PIL import Image
import io

class QwenPolicyClient:
    """
    Client for interacting with the QWEN VLM Policy Server.
    """
    
    def __init__(self, server_url="http://localhost:8001"):
        """
        Initialize the policy client.
        
        Args:
            server_url: URL of the policy server
        """
        self.server_url = server_url
        self.step_count = 0
        
    def check_health(self):
        """Check if the server is healthy and ready."""
        try:
            response = requests.get(f"{self.server_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Health check failed: {e}")
            return None
    
    def reset(self):
        """Reset the step counter."""
        self.step_count = 0
    
    def _encode_observation(self, obs):
        """
        Encode MineRL observation to base64.
        
        Args:
            obs: MineRL observation dictionary with 'pov' key
            
        Returns:
            str: Base64 encoded image
        """
        # Convert observation to PIL Image
        pov = obs['pov']
        image = Image.fromarray(pov.astype(np.uint8))
        
        # Encode to base64
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        return img_str
    
    def get_action(self, obs, task_name, temperature=0.7, max_tokens=512):
        """
        Get action from the policy server.
        
        Args:
            obs: MineRL observation dictionary
            task_name: Name of the BASALT task
            temperature: Sampling temperature for VLM
            max_tokens: Maximum tokens to generate
            
        Returns:
            dict: MineRL-compatible action dictionary
        """
        self.step_count += 1
        
        # Encode observation
        image_base64 = self._encode_observation(obs)
        
        # Prepare request
        payload = {
            "task_name": task_name,
            "image": {
                "data": image_base64
            },
            "step": self.step_count,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            # Make request to server
            response = requests.post(
                f"{self.server_url}/get_action",
                json=payload,
                timeout=30  # 30 second timeout for VLM inference
            )
            response.raise_for_status()
            
            # Parse response
            result = response.json()
            
            # Print reasoning if available
            if 'reasoning' in result:
                print(f"Step {self.step_count} - Reasoning: {result['reasoning']}")
            
            # Convert camera to numpy array if needed
            action = result['action']
            if isinstance(action['camera'], list):
                action['camera'] = np.array(action['camera'], dtype=np.float32)
            
            return action
            
        except requests.exceptions.Timeout:
            print(f"Request timed out at step {self.step_count}")
            return self._get_default_action()
        except Exception as e:
            print(f"Error getting action: {e}")
            return self._get_default_action()
    
    def _get_default_action(self):
        """Return a default 'do nothing' action."""
        return {
            'attack': 0,
            'back': 0,
            'forward': 0,
            'jump': 0,
            'left': 0,
            'right': 0,
            'sneak': 0,
            'sprint': 0,
            'use': 0,
            'drop': 0,
            'inventory': 0,
            'hotbar.1': 0,
            'hotbar.2': 0,
            'hotbar.3': 0,
            'hotbar.4': 0,
            'hotbar.5': 0,
            'hotbar.6': 0,
            'hotbar.7': 0,
            'hotbar.8': 0,
            'hotbar.9': 0,
            'camera': np.array([0.0, 0.0], dtype=np.float32),
            'ESC': 0,
        }


def main():
    """
    Example usage of the QWEN policy client.
    """
    import aicrowd_gym
    import minerl  # Need to import minerl to register environments
    
    # Initialize client
    client = QwenPolicyClient(server_url="http://localhost:8001")
    
    # Check server health
    health = client.check_health()
    if health:
        print("Server Status:", json.dumps(health, indent=2))
    else:
        print("Server is not available!")
        return
    
    # Create environment
    task_name = "MineRLBasaltFindCave-v0"
    env = aicrowd_gym.make(task_name)
    
    print(f"\nRunning agent with QWEN policy server on {task_name}")
    print("="*60)
    
    # Run for 1 episode
    obs = env.reset()
    client.reset()
    
    for step in range(100):  # Run for 100 steps
        # Get action from policy server
        action = client.get_action(obs, task_name)
        
        # Step environment
        obs, reward, done, info = env.step(action)
        
        if done:
            print(f"\nEpisode finished at step {step + 1}")
            break
    
    env.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
