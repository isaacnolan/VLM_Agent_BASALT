"""
Client for QWEN VLM Policy Server

This module provides a client class for interacting with the QWEN policy server.
"""

import requests
import base64
import json
import numpy as np
from PIL import Image
import io
import logging
from collections import deque
from common.models import PolicyRequest, StateActionPair, ImageData
from common.actions import get_default_minerl_action

logger = logging.getLogger(__name__)


class QwenPolicyClient:
    """
    Client for interacting with the QWEN VLM Policy Server.
    """
    
    def __init__(self, server_url="http://localhost:8001", max_history_length=5):
        """
        Initialize the policy client.
        
        Args:
            server_url: URL of the policy server
            max_history_length: Maximum number of state-action pairs to keep in history
        """
        self.server_url = server_url
        self.step_count = 0
        self.episode_data = []  # Store actions and reasoning for each step
        self.max_history_length = max_history_length
        self.hist_context = deque(maxlen=max_history_length)  # History queue of state-action pairs
        
    def check_health(self):
        """Check if the server is healthy and ready."""
        try:
            response = requests.get(f"{self.server_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return None
    
    def reset(self):
        """Reset the step counter, episode data, and history context."""
        self.step_count = 0
        self.episode_data = []
        self.hist_context.clear()
    
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
        
        # Encode current observation
        image_base64 = self._encode_observation(obs)
        
        # Add current observation to history (without action yet)
        current_state = {
            "image": {
                "data": image_base64
            },
            "action": None  # Will be filled in after we get the action
        }
        self.hist_context.append(current_state)
        
        # Prepare request with history
        payload = {
            "task_name": task_name,
            "history": list(self.hist_context),  # Convert deque to list for JSON
            "step": self.step_count,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_history_length": self.max_history_length
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
            
            # Log reasoning if available
            if 'reasoning' in result:
                logger.info(f"Step {self.step_count} - Reasoning: {result['reasoning']}")
            
            # Convert camera to numpy array if needed
            action = result['action']
            if isinstance(action['camera'], list):
                action['camera'] = np.array(action['camera'], dtype=np.float32)
            
            # Update the last history entry with the action taken
            if self.hist_context:
                self.hist_context[-1]["action"] = action.copy()
                # Convert numpy array back to list for storage
                if isinstance(self.hist_context[-1]["action"]['camera'], np.ndarray):
                    self.hist_context[-1]["action"]['camera'] = self.hist_context[-1]["action"]['camera'].tolist()
            
            # Store action and reasoning for later analysis
            self.episode_data.append({
                'step': self.step_count,
                'action': action.copy(),
                'reasoning': result.get('reasoning', 'No reasoning provided')
            })
            
            return action
            
        except requests.exceptions.Timeout:
            logger.warning(f"Request timed out at step {self.step_count}")
            action = get_default_minerl_action()
            action['camera'] = np.array(action['camera'], dtype=np.float32)
            
            # Update history with default action
            if self.hist_context:
                self.hist_context[-1]["action"] = action.copy()
                if isinstance(self.hist_context[-1]["action"]['camera'], np.ndarray):
                    self.hist_context[-1]["action"]['camera'] = self.hist_context[-1]["action"]['camera'].tolist()
            
            self.episode_data.append({
                'step': self.step_count,
                'action': action.copy(),
                'reasoning': 'Request timeout - using default action'
            })
            return action
        except Exception as e:
            logger.error(f"Error getting action: {e}")
            action = get_default_minerl_action()
            action['camera'] = np.array(action['camera'], dtype=np.float32)
            
            # Update history with default action
            if self.hist_context:
                self.hist_context[-1]["action"] = action.copy()
                if isinstance(self.hist_context[-1]["action"]['camera'], np.ndarray):
                    self.hist_context[-1]["action"]['camera'] = self.hist_context[-1]["action"]['camera'].tolist()
            
            self.episode_data.append({
                'step': self.step_count,
                'action': action.copy(),
                'reasoning': f'Error: {str(e)} - using default action'
            })
            return action
    
    def save_episode_data(self, record_dir, episode_num):
        """
        Save episode data (actions and reasoning) to a JSON file.
        
        Args:
            record_dir: Directory to save the data
            episode_num: Episode number for filename
        """
        import os
        
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
        
        logger.info(f"Saved episode data to {filepath}")
