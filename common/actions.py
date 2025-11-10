"""Action manipulation utilities shared by client and server."""
import numpy as np
from typing import Dict, Any


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


def create_minerl_action(action_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a MineRL-formatted action from a dictionary.
    
    Handles partial action dicts by filling in missing keys with defaults.
    
    Args:
        action_dict: Partial or complete action dictionary
        
    Returns:
        Complete MineRL action dictionary
    """
    def to_int_or_zero(val):
        try:
            return int(val)
        except Exception:
            return 0

    # Start with all defaults
    minerl_action = get_default_minerl_action()
    
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


def get_exploration_action() -> Dict[str, Any]:
    """Return a basic exploration action (move forward and look around)."""
    action = get_default_minerl_action()
    action['forward'] = 1
    return action


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
