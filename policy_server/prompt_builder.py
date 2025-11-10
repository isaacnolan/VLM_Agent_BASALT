"""Build prompts for the VLM model."""
import logging
import json
from typing import List, Dict, Any
from PIL import Image

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Handles prompt construction for VLM queries."""
    
    def __init__(self, config_path="config/task_prompts.json"):
        """
        Initialize the prompt builder.
        
        Args:
            config_path: Path to task prompts configuration file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.action_template = self.config.get("action_prompt_template", "")
        self.velocity_guidance = self.config.get("velocity_guidance", "")
    
    def get_task_config(self, task_name: str) -> Dict[str, str]:
        """
        Get configuration for a specific task.
        
        Args:
            task_name: Name of the BASALT task
            
        Returns:
            Dictionary with system and task_description
        """
        return self.config.get(task_name, self.config.get("MineRLBasaltFindCave-v0"))
    
    def build_text_prompt(self, task_name: str, step: int, 
                          history_text: str = None) -> str:
        """
        Build the text prompt for VLM.
        
        Args:
            task_name: Name of the BASALT task
            step: Current step number
            history_text: Optional history summary
            
        Returns:
            Formatted text prompt
        """
        task_config = self.get_task_config(task_name)
        
        if history_text:
            text_prompt = f"""Step {step}

{task_config['task_description']}

Recent History:{history_text}

Current observation is shown in the most recent image.

{self.velocity_guidance}

{self.action_template}"""
        else:
            text_prompt = f"""Step {step}

{task_config['task_description']}

{self.action_template}"""
        
        return text_prompt
    
    def build_messages(self, task_name: str, images: List[Image.Image],
                      text_prompt: str) -> List[Dict]:
        """
        Build the full message structure for VLM.
        
        Args:
            task_name: Name of the BASALT task
            images: List of PIL Images
            text_prompt: Text prompt string
            
        Returns:
            Messages list in format expected by model
        """
        task_config = self.get_task_config(task_name)
        
        # Build content with all images
        content = []
        
        # Add all images to content
        for idx, img in enumerate(images):
            if len(images) > 1:
                # Label images in history
                if idx < len(images) - 1:
                    content.append({
                        "type": "text",
                        "text": f"[Historical Frame {idx + 1}]"
                    })
                else:
                    content.append({
                        "type": "text",
                        "text": "[Current Frame]"
                    })
            content.append({
                "type": "image",
                "image": img,
            })
        
        # Add the main text prompt at the end
        content.append({
            "type": "text",
            "text": text_prompt
        })
        
        messages = [
            {
                "role": "system",
                "content": task_config['system']
            },
            {
                "role": "user",
                "content": content
            }
        ]
        
        return messages
