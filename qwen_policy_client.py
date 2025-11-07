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
import cv2
import os
from argparse import ArgumentParser

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
        self.episode_data = []  # Store actions and reasoning for each step
        
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
        """Reset the step counter and episode data."""
        self.step_count = 0
        self.episode_data = []
    
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
            
            # Store action and reasoning for later analysis
            self.episode_data.append({
                'step': self.step_count,
                'action': action.copy(),
                'reasoning': result.get('reasoning', 'No reasoning provided')
            })
            
            return action
            
        except requests.exceptions.Timeout:
            print(f"Request timed out at step {self.step_count}")
            action = self._get_default_action()
            self.episode_data.append({
                'step': self.step_count,
                'action': action.copy(),
                'reasoning': 'Request timeout - using default action'
            })
            return action
        except Exception as e:
            print(f"Error getting action: {e}")
            action = self._get_default_action()
            self.episode_data.append({
                'step': self.step_count,
                'action': action.copy(),
                'reasoning': f'Error: {str(e)} - using default action'
            })
            return action
    
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
    
    def save_episode_data(self, record_dir, episode_num):
        """
        Save episode data (actions and reasoning) to a JSON file.
        
        Args:
            record_dir: Directory to save the data
            episode_num: Episode number for filename
        """
        if not self.episode_data:
            return
        
        os.makedirs(record_dir, exist_ok=True)
        filepath = os.path.join(record_dir, f"episode_{episode_num:03d}_data.json")
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = []
        for step_data in self.episode_data:
            data = step_data.copy()
            if isinstance(data['action']['camera'], np.ndarray):
                data['action']['camera'] = data['action']['camera'].tolist()
            serializable_data.append(data)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
        print(f"Saved episode data to {filepath}")


def main(task_name, n_episodes=3, max_steps=100, show=False, record_dir=None, server_url="http://localhost:8001"):
    """
    Run QWEN policy client with optional video recording.
    
    Args:
        task_name: MineRL BASALT task name
        n_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        show: Whether to render the environment
        record_dir: Directory to save videos and episode data (None to disable)
        server_url: URL of the QWEN policy server
    """
    import aicrowd_gym
    import minerl  # Need to import minerl to register environments
    import sys
    import traceback
    
    # Initialize client
    client = QwenPolicyClient(server_url=server_url)
    
    # Check server health
    health = client.check_health()
    if health:
        print("Server Status:", json.dumps(health, indent=2))
    else:
        print("Server is not available!")
        return
    
    # Prepare recording directory if requested
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
        print(f"Recording to directory: {record_dir}")
    
    # Create environment
    env = aicrowd_gym.make(task_name)
    
    print(f"\nRunning agent with QWEN policy server on {task_name}")
    print(f"Episodes: {n_episodes}, Max steps per episode: {max_steps}")
    print("="*60)
    
    for ep in range(n_episodes):
        print(f"\n--- Episode {ep + 1}/{n_episodes} ---")
        
        try:
            obs = env.reset()
        except Exception:
            # Provide more context to help debug Malmo/Minecraft startup failures.
            print("Error during env.reset() - dumping diagnostics:", file=sys.stderr)
            traceback.print_exc()
            print("Common causes: Java (OpenJDK) missing, Malmo failed to start, X11/Xvfb not available, or permissions issues.", file=sys.stderr)
            raise
        
        client.reset()
        
        # Create video writer for this episode if requested
        writer = None
        if record_dir and obs is not None and "pov" in obs:
            h, w, _ = obs["pov"].shape
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_path = os.path.join(record_dir, f"episode_{ep:03d}.mp4")
            # OpenCV expects (width, height)
            writer = cv2.VideoWriter(video_path, fourcc, 20.0, (w, h))
            print(f"Recording video to: {video_path}")
        
        try:
            for step in range(max_steps):
                # Get action from policy server
                action = client.get_action(obs, task_name)
                
                # ESC is not part of the predictions model
                action["ESC"] = 0
                
                # Step environment
                obs, reward, done, info = env.step(action)
                
                if show:
                    env.render()
                
                # Write frame from observation if recording
                if writer is not None and obs is not None and "pov" in obs:
                    try:
                        frame = obs["pov"]
                        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        writer.write(frame_bgr)
                    except Exception as e:
                        # best-effort: ignore frame write errors
                        print(f"Warning: Could not write frame {step}: {e}")
                
                if done:
                    print(f"Episode finished at step {step + 1}")
                    break
        
        finally:
            # Always release writer for this episode, even if interrupted
            if writer is not None:
                writer.release()
                print(f"Video saved: episode_{ep:03d}.mp4")
            
            # Save episode data (actions and reasoning)
            if record_dir:
                client.save_episode_data(record_dir, ep)
    
    env.close()
    print("\nDone!")


if __name__ == "__main__":
    parser = ArgumentParser("Run QWEN VLM policy client on MineRL BASALT environment")
    
    parser.add_argument("--task", type=str, default="MineRLBasaltFindCave-v0",
                        help="BASALT task name (default: MineRLBasaltFindCave-v0)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run (default: 3)")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum steps per episode (default: 1000)")
    parser.add_argument("--show", action="store_true",
                        help="Render the environment")
    parser.add_argument("--record-dir", type=str, default=None,
                        help="Directory to save episode videos and data (default: None)")
    parser.add_argument("--server-url", type=str, default="http://localhost:8001",
                        help="URL of the QWEN policy server (default: http://localhost:8001)")
    
    args = parser.parse_args()
    
    main(
        task_name=args.task,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        show=args.show,
        record_dir=args.record_dir,
        server_url=args.server_url
    )
