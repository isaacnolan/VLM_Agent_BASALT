"""
QWEN VLM Agent for MineRL BASALT Tasks

This agent uses the Qwen2-VL-4B model as a policy function to make decisions
based on visual observations from the Minecraft environment.
"""

from argparse import ArgumentParser
import json
import numpy as np
import torch
from PIL import Image
import io
import base64
import os
from datetime import datetime
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import aicrowd_gym
import minerl
from openai_vpt.lib.actions import ActionTransformer

# MineRL action space configuration
ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)

# Task-specific prompts for the VLM
TASK_PROMPTS = {
    "MineRLBasaltFindCave-v0": {
        "system": "You are an expert Minecraft player tasked with finding a cave entrance.",
        "task_description": """Your goal is to locate and identify a cave entrance in Minecraft.
        
Key objectives:
- Explore the environment to find natural cave openings
- Look for dark openings in cliff faces, hillsides, or underground
- Navigate towards visible cave entrances
- Stay aware of your surroundings

Based on the current observation, decide what action to take.""",
        "action_prompt": """Analyze the image and decide the best action:
        
Actions available:
- forward/back: Move forward or backward
- left/right: Strafe left or right  
- jump: Jump
- sneak: Crouch/sneak
- sprint: Sprint
- attack: Break blocks
- camera: Look around (horizontal and vertical angles)

Respond in JSON format:
{
    "reasoning": "Brief explanation of what you see and why you chose this action",
    "action": {
        "forward": 0 or 1,
        "back": 0 or 1,
        "left": 0 or 1,
        "right": 0 or 1,
        "jump": 0 or 1,
        "sneak": 0 or 1,
        "sprint": 0 or 1,
        "attack": 0 or 1,
        "camera": [horizontal_angle, vertical_angle]
    }
}"""
    },
    "MineRLBasaltMakeWaterfall-v0": {
        "system": "You are an expert Minecraft player tasked with creating a beautiful waterfall.",
        "task_description": """Your goal is to create an aesthetically pleasing waterfall in Minecraft.
        
Key objectives:
- Find a suitable location (cliff, elevated terrain)
- Gather necessary materials (water buckets, blocks for shaping)
- Create a flowing waterfall with good visual appeal
- Consider the surrounding environment for aesthetics

Based on the current observation, decide what action to take.""",
        "action_prompt": """Analyze the image and decide the best action:
        
Actions available:
- forward/back: Move forward or backward
- left/right: Strafe left or right
- jump: Jump
- sneak: Crouch/sneak
- sprint: Sprint
- attack: Break blocks
- use: Place blocks or use items
- camera: Look around (horizontal and vertical angles)
- inventory management

Respond in JSON format with your reasoning and chosen action."""
    },
    "MineRLBasaltCreateVillageAnimalPen-v0": {
        "system": "You are an expert Minecraft player tasked with building an animal pen in a village.",
        "task_description": """Your goal is to create a functional and safe animal pen in a village.
        
Key objectives:
- Locate the village and find a suitable spot for the pen
- Gather materials (fences, gates, etc.)
- Build an enclosed area to keep animals safe
- Ensure the pen fits the village aesthetic
- Make it functional with proper gates

Based on the current observation, decide what action to take.""",
        "action_prompt": """Analyze the image and decide the best action:
        
Actions available:
- Movement: forward, back, left, right, jump, sneak, sprint
- Interaction: attack (break), use (place/interact)
- Camera: Look around
- Inventory: Manage items

Respond in JSON format with your reasoning and chosen action."""
    },
    "MineRLBasaltBuildVillageHouse-v0": {
        "system": "You are an expert Minecraft player tasked with building a house in a village.",
        "task_description": """Your goal is to construct a house that fits the village aesthetic.
        
Key objectives:
- Locate the village and understand its building style
- Gather appropriate materials
- Build a house that matches village architecture
- Include proper doors, windows, and roof
- Ensure it integrates well with existing structures

Based on the current observation, decide what action to take.""",
        "action_prompt": """Analyze the image and decide the best action:
        
Actions available:
- Movement: forward, back, left, right, jump, sneak, sprint
- Interaction: attack (break), use (place/interact)
- Camera: Look around
- Inventory: Manage items and hotbar

Respond in JSON format with your reasoning and chosen action."""
    }
}


class QwenVLMAgent:
    """
    Agent that uses Qwen2-VL-4B model to make decisions based on visual observations.
    """
    
    def __init__(self, env, task_name, device="cuda", model_name="Qwen/Qwen2-VL-7B-Instruct", log_file=None):
        """
        Initialize the QWEN VLM Agent.
        
        Args:
            env: MineRL environment
            task_name: Name of the BASALT task
            device: Device to run the model on ('cuda' or 'cpu')
            model_name: Hugging Face model identifier for QWEN VLM
            log_file: Path to log file for VLM responses (optional)
        """
        self.device = device
        self.task_name = task_name
        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)
        
        # Get task-specific prompts
        self.task_config = TASK_PROMPTS.get(task_name, TASK_PROMPTS["MineRLBasaltFindCave-v0"])
        
        # Setup logging
        self.log_file = log_file
        if self.log_file:
            # Create logs directory if it doesn't exist
            log_dir = os.path.dirname(self.log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            
            # Initialize log file with header
            with open(self.log_file, 'w') as f:
                f.write(f"VLM Response Log - {task_name}\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
            print(f"Logging VLM responses to: {self.log_file}")
        
        # Load QWEN VLM model
        print(f"Loading QWEN VLM model: {model_name}")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Conversation history for context
        self.conversation_history = []
        self.step_count = 0
        self.episode_count = 0
        
        print(f"QWEN VLM Agent initialized for task: {task_name}")
    
    def reset(self):
        """Reset the agent's conversation history."""
        self.conversation_history = []
        self.step_count = 0
        self.episode_count += 1
        
        # Log episode start
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"Episode {self.episode_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"{'='*80}\n\n")
    
    def _log_vlm_response(self, step, prompt, response_text, action):
        """
        Log VLM response to file.
        
        Args:
            step: Current step number
            prompt: The prompt sent to the VLM
            response_text: The raw response from the VLM
            action: The parsed MineRL action
        """
        if not self.log_file:
            return
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(f"--- Step {step} [{datetime.now().strftime('%H:%M:%S')}] ---\n")
                f.write(f"\nPrompt:\n{prompt}\n")
                f.write(f"\nVLM Response:\n{response_text}\n")
                f.write(f"\nParsed Action:\n")
                
                # Log non-zero actions for readability
                action_summary = []
                for key, value in action.items():
                    if key == 'camera':
                        if not np.allclose(value, 0):
                            action_summary.append(f"  {key}: [{value[0]:.2f}, {value[1]:.2f}]")
                    elif value != 0:
                        action_summary.append(f"  {key}: {value}")
                
                if action_summary:
                    f.write("\n".join(action_summary) + "\n")
                else:
                    f.write("  (no action)\n")
                
                f.write("\n" + "-"*80 + "\n\n")
        except Exception as e:
            print(f"Warning: Failed to log VLM response: {e}")
    
    def _observation_to_image(self, obs):
        """
        Convert MineRL observation to PIL Image.
        
        Args:
            obs: MineRL observation dictionary with 'pov' key
            
        Returns:
            PIL.Image: The observation as a PIL Image
        """
        # obs['pov'] is a numpy array of shape (height, width, 3) with RGB values
        pov = obs['pov']
        
        # Convert to PIL Image
        image = Image.fromarray(pov.astype(np.uint8))
        
        return image
    
    def _create_prompt(self, step_count):
        """
        Create the text prompt for the VLM based on the current step.
        
        Args:
            step_count: Current step number in the episode
            
        Returns:
            str: Formatted prompt for the VLM
        """
        prompt = f"""Step {step_count}

{self.task_config['task_description']}

{self.task_config['action_prompt']}

Keep your reasoning concise and focused on the immediate next action."""
        
        return prompt
    
    def _parse_vlm_response(self, response_text):
        """
        Parse the VLM's JSON response into a MineRL action.
        
        Args:
            response_text: Text response from the VLM
            
        Returns:
            dict: MineRL-compatible action dictionary
        """
        try:
            # Try to extract JSON from the response
            # The model might return text before/after the JSON
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                response_json = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in response")
            
            # Extract action from response
            action_dict = response_json.get('action', {})
            
            # Create MineRL action format
            minerl_action = {
                'attack': int(action_dict.get('attack', 0)),
                'back': int(action_dict.get('back', 0)),
                'forward': int(action_dict.get('forward', 0)),
                'jump': int(action_dict.get('jump', 0)),
                'left': int(action_dict.get('left', 0)),
                'right': int(action_dict.get('right', 0)),
                'sneak': int(action_dict.get('sneak', 0)),
                'sprint': int(action_dict.get('sprint', 0)),
                'use': int(action_dict.get('use', 0)),
                'drop': int(action_dict.get('drop', 0)),
                'inventory': int(action_dict.get('inventory', 0)),
                'hotbar.1': int(action_dict.get('hotbar.1', 0)),
                'hotbar.2': int(action_dict.get('hotbar.2', 0)),
                'hotbar.3': int(action_dict.get('hotbar.3', 0)),
                'hotbar.4': int(action_dict.get('hotbar.4', 0)),
                'hotbar.5': int(action_dict.get('hotbar.5', 0)),
                'hotbar.6': int(action_dict.get('hotbar.6', 0)),
                'hotbar.7': int(action_dict.get('hotbar.7', 0)),
                'hotbar.8': int(action_dict.get('hotbar.8', 0)),
                'hotbar.9': int(action_dict.get('hotbar.9', 0)),
                'camera': np.array(action_dict.get('camera', [0.0, 0.0]), dtype=np.float32),
                'ESC': 0,
            }
            
            # Log the reasoning if available
            if 'reasoning' in response_json:
                print(f"  Reasoning: {response_json['reasoning']}")
            
            return minerl_action
            
        except Exception as e:
            print(f"Error parsing VLM response: {e}")
            print(f"Response text: {response_text}")
            
            # Return a safe default action (do nothing)
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
    
    def get_action(self, obs):
        """
        Get action from the VLM based on the current observation.
        
        Args:
            obs: MineRL observation dictionary
            
        Returns:
            dict: MineRL-compatible action
        """
        self.step_count += 1
        
        # Convert observation to image
        image = self._observation_to_image(obs)
        
        # Create prompt
        text_prompt = self._create_prompt(self.step_count)
        
        # Prepare messages for the VLM
        messages = [
            {
                "role": "system",
                "content": self.task_config['system']
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": text_prompt
                    }
                ]
            }
        ]
        
        # Prepare inputs for the model
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(self.device)
        
        # Generate response
        print(f"Step {self.step_count}: Querying VLM...")
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response_text = self.processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        print(f"  VLM Response: {response_text[:200]}...")
        
        # Parse response into action
        action = self._parse_vlm_response(response_text)
        
        # Log the VLM response
        self._log_vlm_response(self.step_count, text_prompt, response_text, action)
        
        return action


def main(env, task_name, n_episodes=3, max_steps=int(1e9), show=False, device="cuda", log_file=None):
    """
    Main function to run the QWEN VLM agent on a MineRL BASALT task.
    
    Args:
        env: Environment name
        task_name: Name of the BASALT task
        n_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        show: Whether to render the environment
        device: Device to run the model on
        log_file: Path to log file for VLM responses (optional)
    """
    # Using aicrowd_gym is important! Your submission will not work otherwise
    env = aicrowd_gym.make(task_name)
    
    # Initialize QWEN VLM agent
    agent = QwenVLMAgent(env, task_name, device=device, log_file=log_file)
    
    print(f"\n{'='*60}")
    print(f"Running QWEN VLM Agent on {task_name}")
    print(f"Episodes: {n_episodes}, Max steps: {max_steps}")
    print(f"{'='*60}\n")
    
    for episode in range(n_episodes):
        print(f"\n--- Episode {episode + 1}/{n_episodes} ---")
        obs = env.reset()
        agent.reset()
        
        total_reward = 0
        
        for step in range(max_steps):
            # Get action from VLM
            action = agent.get_action(obs)
            
            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            if show:
                env.render()
            
            if done:
                print(f"Episode finished at step {step + 1}")
                break
        
        print(f"Episode {episode + 1} total reward: {total_reward}")
    
    env.close()
    print(f"\n{'='*60}")
    print("QWEN VLM Agent run complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = ArgumentParser("Run QWEN VLM agent on MineRL BASALT environment")
    
    parser.add_argument(
        "--env", 
        type=str, 
        required=True,
        choices=[
            "MineRLBasaltFindCave-v0",
            "MineRLBasaltMakeWaterfall-v0", 
            "MineRLBasaltCreateVillageAnimalPen-v0",
            "MineRLBasaltBuildVillageHouse-v0"
        ],
        help="MineRL BASALT environment name"
    )
    parser.add_argument(
        "--n_episodes", 
        type=int, 
        default=3, 
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--max_steps", 
        type=int, 
        default=int(1e9), 
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--show", 
        action="store_true", 
        help="Render the environment"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run the model on"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Hugging Face model name for QWEN VLM (use Qwen2-VL-2B-Instruct for 4B param model)"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to log file for VLM responses (default: logs/vlm_responses_<timestamp>.txt)"
    )
    
    args = parser.parse_args()
    
    # Create default log file if not specified
    log_file = args.log_file
    if log_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"logs/vlm_responses_{timestamp}.txt"
    
    main(
        env=args.env,
        task_name=args.env,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        show=args.show,
        device=args.device,
        log_file=log_file
    )
