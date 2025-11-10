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

ACTION_PROMPT_TEMPLATE = """Analyze the image and decide the best action:

Actions available:
- forward/back: Move forward or backward (0 or 1)
- left/right: Strafe left or right (0 or 1)
- jump: Jump (0 or 1)
- sneak: Crouch/sneak (0 or 1)
- sprint: Sprint (0 or 1)
- attack: Break blocks (0 or 1)
- use: Place blocks or use items (0 or 1)
- drop: Drop items (0 or 1)
- inventory: Open inventory (0 or 1)
- camera: Look around [horizontal_angle, vertical_angle] in degrees
- hotbar.1 through hotbar.9: Select hotbar slot (0 or 1)

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
        "use": 0 or 1,
        "drop": 0 or 1,
        "inventory": 0 or 1,
        "camera": [horizontal_angle, vertical_angle],
        "hotbar.1": 0 or 1,
        "hotbar.2": 0 or 1,
        "hotbar.3": 0 or 1,
        "hotbar.4": 0 or 1,
        "hotbar.5": 0 or 1,
        "hotbar.6": 0 or 1,
        "hotbar.7": 0 or 1,
        "hotbar.8": 0 or 1,
        "hotbar.9": 0 or 1
    }
}

Keep your reasoning concise and focused on the immediate next action."""


class ImageData(BaseModel):
    data: str  # base64 encoded image

class StateActionPair(BaseModel):
    """Represents a historical state-action pair"""
    image: ImageData
    action: Optional[Dict[str, Any]] = None  # The action taken from this state
    
class PolicyRequest(BaseModel):
    task_name: str
    image: Optional[ImageData] = None  # Current image (for backward compatibility)
    history: Optional[List[StateActionPair]] = None  # History of state-action pairs
    step: Optional[int] = 0
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    max_history_length: Optional[int] = 5  # Maximum number of historical frames to include

class ActionResponse(BaseModel):
    action: Dict[str, Any]
    reasoning: str
    step: int

def summarize_action(action: Dict[str, Any]) -> str:
    """
    Create a human-readable summary of an action.
    
    Args:
        action: MineRL action dictionary
        
    Returns:
        String summary of the action
    """
    parts = []
    
    # Movement actions
    if action.get('forward'):
        parts.append("moved forward")
    if action.get('back'):
        parts.append("moved back")
    if action.get('left'):
        parts.append("strafed left")
    if action.get('right'):
        parts.append("strafed right")
    if action.get('jump'):
        parts.append("jumped")
    if action.get('sneak'):
        parts.append("sneaked")
    if action.get('sprint'):
        parts.append("sprinted")
    
    # Interaction actions
    if action.get('attack'):
        parts.append("attacked/mined")
    if action.get('use'):
        parts.append("used item/placed block")
    if action.get('drop'):
        parts.append("dropped item")
    if action.get('inventory'):
        parts.append("opened inventory")
    
    # Hotbar selection
    for i in range(1, 10):
        if action.get(f'hotbar.{i}'):
            parts.append(f"selected hotbar slot {i}")
    
    # Camera movement
    camera = action.get('camera', [0.0, 0.0])
    if isinstance(camera, (list, tuple)) and len(camera) >= 2:
        h, v = camera[0], camera[1]
        if abs(h) > 0.5 or abs(v) > 0.5:
            direction = []
            if h > 0.5:
                direction.append("right")
            elif h < -0.5:
                direction.append("left")
            if v > 0.5:
                direction.append("down")
            elif v < -0.5:
                direction.append("up")
            if direction:
                parts.append(f"looked {' and '.join(direction)}")
    
    if not parts:
        return "no action"
    
    return ", ".join(parts)

@app.on_event("startup")
async def load_model():
    """Load the QWEN VLM model on server startup."""
    global model, processor, device
    
    logger.info("Loading QWEN VLM model...")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model - using 2B model for faster inference
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        
        processor = AutoProcessor.from_pretrained(model_name)
        
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
        "model": "Qwen2-VL-2B-Instruct",
        "device": device,
        "ready": model is not None and processor is not None
    }

@app.post("/get_action", response_model=ActionResponse)
async def get_action(request: PolicyRequest):
    """
    Get an action from the QWEN VLM policy given an observation and optional history.
    
    Args:
        request: PolicyRequest containing task name, current image or history of state-action pairs, and optional parameters
        
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
        # Process history or single image
        images = []
        history_text = ""
        
        if request.history:
            # Use history mode
            logger.info(f"Processing history with {len(request.history)} state-action pairs")
            
            # Limit history length to avoid token overflow
            max_len = request.max_history_length or 5
            history_subset = request.history[-max_len:]
            
            # Decode all historical images
            for idx, state_action in enumerate(history_subset):
                try:
                    image_bytes = base64.b64decode(state_action.image.data)
                    img = Image.open(io.BytesIO(image_bytes))
                    images.append(img)
                    
                    # Build history description
                    if state_action.action:
                        # Summarize the action taken
                        action_summary = summarize_action(state_action.action)
                        history_text += f"\nStep {request.step - len(history_subset) + idx}: {action_summary}"
                    else:
                        history_text += f"\nStep {request.step - len(history_subset) + idx}: (initial state)"
                        
                except Exception as e:
                    logger.error(f"Failed to decode history image {idx}: {e}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid image data in history index {idx}: {str(e)}"
                    )
            
            logger.info(f"Decoded {len(images)} images from history")
            
        elif request.image:
            # Backward compatibility: single image mode
            logger.info("Processing single image (no history)")
            try:
                image_bytes = base64.b64decode(request.image.data)
                image = Image.open(io.BytesIO(image_bytes))
                images = [image]
                logger.info(f"Decoded image: {image.size}, mode: {image.mode}")
            except Exception as e:
                logger.error(f"Failed to decode image: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid image data: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="Either 'image' or 'history' must be provided"
            )
        
        # Get task-specific prompts
        task_config = TASK_PROMPTS.get(
            request.task_name, 
            TASK_PROMPTS["MineRLBasaltFindCave-v0"]
        )
        
        # Construct prompt with history context
        if history_text:
            text_prompt = f"""Step {request.step}

{task_config['task_description']}

Recent History:{history_text}

Current observation is shown in the most recent image.

IMPORTANT - Velocity and Obstacle Detection:
You can estimate velocity from the history of images by observing how the visual scene changes between frames. If you have been taking movement actions (like forward, sprint) but the visual scene shows little to no change between consecutive frames, this indicates a HIGH PROBABILITY of an obstacle blocking your path. In such cases:
- The lack of visual motion suggests you are stuck or blocked
- You should IMMEDIATELY take action to maneuver around the obstacle
- Consider: turning the camera to look around, strafing left/right, jumping, or backing up
- Avoid continuing the same forward movement that led to being stuck

{ACTION_PROMPT_TEMPLATE}"""
        else:
            text_prompt = f"""Step {request.step}

{task_config['task_description']}

{ACTION_PROMPT_TEMPLATE}"""
        
        # Prepare messages for the VLM
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
        logger.info(f"Generating action from VLM (with {len(images)} image(s))...")
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
        
        # Parse the response
        action_dict, reasoning = parse_vlm_response(response_text)
        
        return ActionResponse(
            action=action_dict,
            reasoning=reasoning,
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
