"""
FastAPI Server for QWEN VLM Policy Function

This server provides an API endpoint for getting Minecraft actions from the QWEN VLM model.
It processes visual observations and returns actions in MineRL format.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import torch
from PIL import Image
import io
import base64
import json
import logging
import numpy as np
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="QWEN VLM Policy Server", version="1.0.0")

# Global model and processor (loaded once on startup)
model = None
processor = None
device = None

# Task-specific prompts
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
    },
    "MineRLBasaltCreateVillageAnimalPen-v0": {
        "system": "You are an expert Minecraft player tasked with building an animal pen in a village.",
        "task_description": """Your goal is to create a functional and safe animal pen in a village.
        
Key objectives:
- Locate the village and find a suitable spot for the pen
- Gather materials (fences, gates, etc.)
- Build an enclosed area to keep animals safe
- Ensure the pen fits the village aesthetic

Based on the current observation, decide what action to take.""",
    },
    "MineRLBasaltBuildVillageHouse-v0": {
        "system": "You are an expert Minecraft player tasked with building a house in a village.",
        "task_description": """Your goal is to construct a house that fits the village aesthetic.
        
Key objectives:
- Locate the village and understand its building style
- Gather appropriate materials
- Build a house that matches village architecture
- Include proper doors, windows, and roof

Based on the current observation, decide what action to take.""",
    }
}

ACTION_PROMPT_TEMPLATE = """You are controlling a Minecraft character. Plan the next {n_steps} actions.

Available actions (use 0 or 1 for each):
forward, back, left, right, jump, sneak, sprint, attack, use, drop, inventory
camera: [horizontal_degrees, vertical_degrees]
hotbar.1 through hotbar.9: select inventory slot

Respond with valid JSON in this EXACT format (start with {{ not [):
{{
  "reasoning": "your overall plan",
  "actions": [
    {{"step_reasoning": "step 1 plan", "action": {{"forward": 1, "camera": [0, 0]}}}},
    {{"step_reasoning": "step 2 plan", "action": {{"forward": 1, "camera": [0, 0]}}}}
  ]
}}

CRITICAL: 
- Start with {{ not [
- Include "reasoning" field
- Include "actions" array with {n_steps} items
- No extra text before or after the JSON"""


class ImageData(BaseModel):
    data: str  # base64 encoded image
    
class PolicyRequest(BaseModel):
    task_name: str
    image: ImageData
    step: Optional[int] = 0
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    n_steps: Optional[int] = 5  # Number of steps to plan ahead

class ActionResponse(BaseModel):
    actions: List[Dict[str, Any]]  # List of actions for multiple steps
    reasoning: str
    step_reasonings: List[str]  # Reasoning for each step
    step: int

@app.on_event("startup")
async def load_model():
    """Load the QWEN VLM model on server startup."""
    global model, processor, device
    
    logger.info("Loading QWEN VLM model...")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model - using 7B model for better performance
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    
    try:
        # Clear CUDA cache if available
        if device == "cuda":
            torch.cuda.empty_cache()
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        
        logger.info(f"Successfully loaded {model_name}")
        logger.info("QWEN VLM Policy Server ready!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "QWEN VLM Policy Server",
        "model_loaded": model is not None,
        "device": device
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model": "Qwen2-VL-7B-Instruct",
        "device": device,
        "ready": model is not None and processor is not None
    }

@app.post("/get_action", response_model=ActionResponse)
async def get_action(request: PolicyRequest):
    """
    Get an action from the QWEN VLM policy given an observation.
    
    Args:
        request: PolicyRequest containing task name, base64 encoded image, and optional parameters
        
    Returns:
        ActionResponse with the predicted action and reasoning
    """
    global model, processor, device
    
    if model is None or processor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please wait for server startup to complete."
        )
    
    logger.info(f"Received policy request for task: {request.task_name}, step: {request.step}")
    
    try:
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(request.image.data)
            image = Image.open(io.BytesIO(image_bytes))
            logger.info(f"Decoded image: {image.size}, mode: {image.mode}")
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image data: {str(e)}"
            )
        
        # Get task-specific prompts
        task_config = TASK_PROMPTS.get(
            request.task_name, 
            TASK_PROMPTS["MineRLBasaltFindCave-v0"]
        )
        
        # Construct prompt with n_steps
        n_steps = request.n_steps if request.n_steps > 0 else 5
        text_prompt = f"""Step {request.step}

{task_config['task_description']}

{ACTION_PROMPT_TEMPLATE.format(n_steps=n_steps)}"""
        
        # Prepare messages for the VLM
        messages = [
            {
                "role": "system",
                "content": task_config['system']
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
        
        # Process for model input
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        inputs = inputs.to(device)
        
        # Generate response
        logger.info("Generating action from VLM...")
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=0.9,
                do_sample=True,
            )
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response_text = processor.batch_decode(
            generated_ids_trimmed, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        logger.info(f"VLM Response: {response_text[:200]}...")
        
        # Parse the response for multiple steps
        actions_list, reasoning, step_reasonings = parse_vlm_response_multi_step(response_text, request.n_steps)
        
        return ActionResponse(
            actions=actions_list,
            reasoning=reasoning,
            step_reasonings=step_reasonings,
            step=request.step
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during action generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error during action generation: {str(e)}"
        )

def parse_vlm_response(response_text: str):
    """
    Parse the VLM's JSON response into action dictionary and reasoning.
    
    Args:
        response_text: Text response from the VLM
        
    Returns:
        Tuple of (action_dict, reasoning)
    """
    try:
        # Try to extract JSON from the response
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in response")

        raw_json = json_match.group()
        logger.info(f"Raw JSON extracted from VLM:\n{raw_json}")

        # Clean up common non-JSON tokens that the model sometimes emits
        clean = raw_json
        # Replace constructs like: "forward": 0 or 1  -> choose 1 (model usually means the action is possible)
        clean = re.sub(r'(?P<key>"[a-zA-Z0-9_.]+"\s*:\s*)(?:0\s*or\s*1|1\s*or\s*0)', r"\g<key>1", clean)
        # Replace boolean-like words (True/False) with JSON booleans
        clean = re.sub(r"\bTrue\b", "true", clean)
        clean = re.sub(r"\bFalse\b", "false", clean)
        # Replace single quotes with double quotes (if any)
        clean = clean.replace("\'", '"')
        # Remove trailing commas before closing braces/brackets
        clean = re.sub(r",\s*(\}|\])", r"\1", clean)

        logger.info(f"Cleaned JSON to parse:\n{clean}")

        response_json = json.loads(clean)

        # Extract action and reasoning
        action_dict = response_json.get('action', {})
        reasoning = response_json.get('reasoning', 'No reasoning provided')

        # Create MineRL action format with all required fields
        def to_int_or_zero(val):
            try:
                return int(val)
            except Exception:
                # if val is a list/tuple (camera), handled elsewhere
                return 0

        minerl_action = {
            'attack': to_int_or_zero(action_dict.get('attack', 0)),
            'back': to_int_or_zero(action_dict.get('back', 0)),
            'forward': to_int_or_zero(action_dict.get('forward', 0)),
            'jump': to_int_or_zero(action_dict.get('jump', 0)),
            'left': to_int_or_zero(action_dict.get('left', 0)),
            'right': to_int_or_zero(action_dict.get('right', 0)),
            'sneak': to_int_or_zero(action_dict.get('sneak', 0)),
            'sprint': to_int_or_zero(action_dict.get('sprint', 0)),
            'use': to_int_or_zero(action_dict.get('use', 0)),
            'drop': to_int_or_zero(action_dict.get('drop', 0)),
            'inventory': to_int_or_zero(action_dict.get('inventory', 0)),
            'hotbar.1': to_int_or_zero(action_dict.get('hotbar.1', 0)),
            'hotbar.2': to_int_or_zero(action_dict.get('hotbar.2', 0)),
            'hotbar.3': to_int_or_zero(action_dict.get('hotbar.3', 0)),
            'hotbar.4': to_int_or_zero(action_dict.get('hotbar.4', 0)),
            'hotbar.5': to_int_or_zero(action_dict.get('hotbar.5', 0)),
            'hotbar.6': to_int_or_zero(action_dict.get('hotbar.6', 0)),
            'hotbar.7': to_int_or_zero(action_dict.get('hotbar.7', 0)),
            'hotbar.8': to_int_or_zero(action_dict.get('hotbar.8', 0)),
            'hotbar.9': to_int_or_zero(action_dict.get('hotbar.9', 0)),
            'camera': action_dict.get('camera', [0.0, 0.0]),
            'ESC': 0,
        }

        # Ensure camera is numeric
        cam = minerl_action['camera']
        try:
            if isinstance(cam, (list, tuple)):
                minerl_action['camera'] = [float(cam[0]), float(cam[1])]
            else:
                minerl_action['camera'] = [0.0, 0.0]
        except Exception:
            minerl_action['camera'] = [0.0, 0.0]

        return minerl_action, reasoning

    except Exception as e:
        logger.error(f"Error parsing VLM response: {e}")
        logger.error(f"Response text: {response_text}")

        # Return a safe default action
        default_action = {
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
            'camera': [0.0, 0.0],
            'ESC': 0,
        }

        return default_action, f"Error parsing response: {str(e)}"


def parse_vlm_response_multi_step(response_text: str, n_steps: int):
    """
    Parse the VLM's JSON response for multiple steps into action list and reasoning.
    
    Args:
        response_text: Text response from the VLM
        n_steps: Expected number of steps
        
    Returns:
        Tuple of (actions_list, overall_reasoning, step_reasonings)
    """
    try:
        # Try to extract JSON from the response
        import re
        
        # First try to find JSON object
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        
        # If not found, try to find JSON array (VLM sometimes returns array directly)
        if not json_match:
            json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON found in response")
            is_array = True
        else:
            is_array = False

        raw_json = json_match.group()
        logger.info(f"Raw JSON extracted from VLM (is_array={is_array}):\n{raw_json[:500]}...")

        # Clean up common non-JSON tokens
        clean = raw_json
        clean = re.sub(r'(?P<key>"[a-zA-Z0-9_.]+"\s*:\s*)(?:0\s*or\s*1|1\s*or\s*0)', r"\g<key>1", clean)
        clean = re.sub(r"\bTrue\b", "true", clean)
        clean = re.sub(r"\bFalse\b", "false", clean)
        clean = clean.replace("\'", '"')
        # Remove trailing commas before closing braces/brackets
        clean = re.sub(r",\s*(\}|\])", r"\1", clean)
        # Remove duplicate commas
        clean = re.sub(r",\s*,", ",", clean)
        # Fix missing commas between properties (common VLM mistake)
        clean = re.sub(r'([0-9\]\}])\s*\n\s*"', r'\1,\n"', clean)
        # Remove comments (sometimes VLMs add them)
        clean = re.sub(r'//.*$', '', clean, flags=re.MULTILINE)

        logger.info(f"Cleaned JSON to parse:\n{clean[:500]}...")

        try:
            response_json = json.loads(clean)
        except json.JSONDecodeError as json_err:
            logger.error(f"JSON decode error: {json_err}")
            logger.error(f"Failed JSON (first 1000 chars):\n{clean[:1000]}")
            
            # Try even more aggressive cleaning
            # Extract just the actions array if possible
            actions_match = re.search(r'"actions"\s*:\s*\[(.*?)\]', clean, re.DOTALL)
            if actions_match:
                logger.info("Attempting to extract just the actions array")
                actions_str = "[" + actions_match.group(1) + "]"
                try:
                    actions_array = json.loads(actions_str)
                    response_json = {"actions": actions_array, "reasoning": "Partially recovered from malformed JSON"}
                except:
                    raise json_err
            else:
                raise json_err

        # Handle case where VLM returns array directly instead of object with "actions" key
        if is_array and isinstance(response_json, list):
            logger.info("VLM returned array directly, wrapping in proper structure")
            actions_data = response_json
            overall_reasoning = "Actions returned as array"
        else:
            # Extract overall reasoning
            overall_reasoning = response_json.get('reasoning', 'No reasoning provided')
            
            # Extract actions list
            actions_data = response_json.get('actions', [])
        
        if not actions_data:
            # Fallback: maybe it's single-step format
            if 'action' in response_json:
                actions_data = [{'action': response_json['action'], 'step_reasoning': overall_reasoning}]
            else:
                logger.warning(f"No 'actions' or 'action' found in response JSON: {response_json}")
                # Try to find any action-like data
                for key in response_json:
                    if isinstance(response_json[key], dict) and 'forward' in response_json[key]:
                        actions_data = [{'action': response_json[key], 'step_reasoning': overall_reasoning}]
                        logger.info(f"Found action data under key '{key}'")
                        break
        
        # Parse each action
        actions_list = []
        step_reasonings = []
        
        for i, action_data in enumerate(actions_data[:n_steps]):  # Limit to n_steps
            # Handle both dict with 'action' key and direct action dict
            if isinstance(action_data, dict):
                if 'action' in action_data:
                    action_dict = action_data.get('action', {})
                    step_reasoning = action_data.get('step_reasoning', f'Step {i+1}')
                else:
                    # Maybe the whole thing is an action
                    action_dict = action_data
                    step_reasoning = f'Step {i+1}'
            else:
                logger.warning(f"Unexpected action_data type: {type(action_data)}")
                action_dict = {}
                step_reasoning = f'Step {i+1} - error'
            
            step_reasonings.append(step_reasoning)
            
            # Create MineRL action format
            minerl_action = create_minerl_action(action_dict)
            actions_list.append(minerl_action)
        
        # If we got fewer actions than requested, repeat the last action
        while len(actions_list) < n_steps:
            if actions_list:
                actions_list.append(actions_list[-1].copy())
                step_reasonings.append("Continuing previous action")
            else:
                # No valid actions found at all - create exploration actions
                logger.warning("No valid actions parsed, creating default exploration actions")
                actions_list.append(get_exploration_action())
                step_reasonings.append("Default exploration action")
        
        logger.info(f"Successfully parsed {len(actions_list)} actions")
        return actions_list, overall_reasoning, step_reasonings

    except Exception as e:
        logger.error(f"Error parsing multi-step VLM response: {e}")
        logger.error(f"Response text: {response_text}")

        # Return default actions for all steps
        default_action = get_default_minerl_action()
        actions_list = [default_action.copy() for _ in range(n_steps)]
        step_reasonings = [f"Error parsing response: {str(e)}"] * n_steps
        
        return actions_list, f"Error parsing response: {str(e)}", step_reasonings


def create_minerl_action(action_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Create a MineRL-formatted action from a dictionary.
    
    Handles partial action dicts by filling in missing keys with defaults.
    """
    def to_int_or_zero(val):
        try:
            return int(val)
        except Exception:
            return 0

    # Start with all defaults
    minerl_action = {
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
        'camera': [0.0, 0.0],
        'ESC': 0,
    }
    
    # Update with provided values
    if not action_dict:
        return minerl_action
    
    for key in minerl_action.keys():
        if key == 'camera':
            cam = action_dict.get('camera', [0.0, 0.0])
            try:
                if isinstance(cam, (list, tuple)) and len(cam) >= 2:
                    minerl_action['camera'] = [float(cam[0]), float(cam[1])]
                else:
                    minerl_action['camera'] = [0.0, 0.0]
            except:
                minerl_action['camera'] = [0.0, 0.0]
        elif key in action_dict:
            minerl_action[key] = to_int_or_zero(action_dict[key])

    return minerl_action


def get_default_minerl_action() -> Dict[str, Any]:
    """Return a default 'do nothing' MineRL action."""
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
        'camera': [0.0, 0.0],
        'ESC': 0,
    }


def get_exploration_action() -> Dict[str, Any]:
    """Return a basic exploration action (move forward and look around)."""
    return {
        'attack': 0,
        'back': 0,
        'forward': 1,
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
        'camera': [0.0, 0.0],
        'ESC': 0,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
