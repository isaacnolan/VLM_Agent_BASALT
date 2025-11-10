# History-Enabled Policy Server API

## Overview

The QWEN VLM Policy Server now supports providing a history of state-action pairs instead of just a single observation. This allows the model to have temporal context about previous observations and actions, enabling better decision-making.

## API Changes

### Request Format

#### New Classes

```python
class StateActionPair(BaseModel):
    """Represents a historical state-action pair"""
    image: ImageData  # Base64 encoded image
    action: Optional[Dict[str, Any]] = None  # The action taken from this state

class PolicyRequest(BaseModel):
    task_name: str
    image: Optional[ImageData] = None  # For backward compatibility
    history: Optional[List[StateActionPair]] = None  # New: history support
    step: Optional[int] = 0
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    max_history_length: Optional[int] = 5  # Maximum historical frames to include
```

### Usage Modes

#### Mode 1: Single Image (Backward Compatible)

Send a single current observation without history:

```python
request = {
    "task_name": "MineRLBasaltFindCave-v0",
    "image": {
        "data": "<base64_encoded_image>"
    },
    "step": 5
}
```

#### Mode 2: History with State-Action Pairs

Send a sequence of observations with the actions taken at each step:

```python
request = {
    "task_name": "MineRLBasaltFindCave-v0",
    "history": [
        {
            "image": {"data": "<base64_encoded_image_t0>"},
            "action": None  # Initial state, no previous action
        },
        {
            "image": {"data": "<base64_encoded_image_t1>"},
            "action": {
                "forward": 1,
                "camera": [0.0, 0.0],
                # ... other action fields
            }
        },
        {
            "image": {"data": "<base64_encoded_image_t2>"},
            "action": {
                "forward": 1,
                "camera": [5.0, -2.0],
                # ... other action fields
            }
        }
    ],
    "step": 2,
    "max_history_length": 5  # Optional: limit context window
}
```

## Key Features

### 1. History Length Management

The `max_history_length` parameter controls how many historical frames are sent to the VLM:
- Default: 5 frames
- If history is longer, only the most recent N frames are used
- Helps manage token limits and memory

### 2. Action Summarization

Actions in the history are automatically summarized in the prompt sent to the VLM. For example:

```
Recent History:
Step 3: moved forward
Step 4: moved forward, looked right and up
Step 5: attacked/mined, moved forward
```

This gives the model a text description of what happened, in addition to the visual observations.

### 3. Multi-Image Context

All historical images are included in the vision input to the model:
- Images are labeled as "[Historical Frame N]" and "[Current Frame]"
- The VLM can see the temporal sequence visually
- Enables understanding of movement, changes, and progress

## Example Integration

### In a MineRL Agent

```python
class HistoryAwareAgent:
    def __init__(self, server_url="http://localhost:8001"):
        self.server_url = server_url
        self.history = []
        self.max_history = 5
    
    def reset(self):
        """Reset history for new episode"""
        self.history = []
    
    def get_action(self, observation):
        """Get action with history context"""
        # Encode current observation
        current_image = encode_image(observation['pov'])
        
        # Add to history (without action yet)
        self.history.append({
            "image": {"data": current_image},
            "action": None
        })
        
        # Request action from server
        response = requests.post(
            f"{self.server_url}/get_action",
            json={
                "task_name": "MineRLBasaltFindCave-v0",
                "history": self.history[-self.max_history:],
                "step": len(self.history)
            }
        )
        
        action = response.json()['action']
        
        # Update last history entry with the action taken
        self.history[-1]['action'] = action
        
        return action
```

## Benefits

1. **Temporal Awareness**: Model can see how the environment changes over time
2. **Action Context**: Model knows what actions were recently taken
3. **Better Planning**: Can avoid repeating failed actions or continue successful strategies
4. **Progress Tracking**: Model can gauge progress toward goals

## Performance Considerations

- More images = more tokens = slower inference
- Recommend `max_history_length` of 3-5 for real-time applications
- Can use longer history (10+) for offline analysis or slower tasks
- Images are processed in parallel by the vision encoder

## Migration Guide

### From Single Image to History

**Before:**
```python
response = requests.post(url, json={
    "task_name": task,
    "image": {"data": encoded_image},
    "step": step
})
```

**After (minimal change):**
```python
# Build history list
history = [{"image": {"data": encoded_image}, "action": last_action}]

response = requests.post(url, json={
    "task_name": task,
    "history": history,
    "step": step
})
```

The old single-image format still works for backward compatibility!
