# Client History Update - Summary

## Overview
The `qwen_policy_client.py` has been updated to maintain and send a history of state-action pairs to the policy server, enabling temporal context for better decision-making.

## Key Changes

### 1. Added History Queue (`hist_context`)
- Uses Python's `deque` with automatic size limiting
- Stores state-action pairs with base64-encoded images
- Automatically removes oldest entries when max length is reached

### 2. Modified `__init__` Method
```python
def __init__(self, server_url="http://localhost:8001", max_history_length=5):
    # ...
    self.max_history_length = max_history_length
    self.hist_context = deque(maxlen=max_history_length)
```

### 3. Updated `reset()` Method
- Now clears the history queue when starting a new episode
- Ensures clean state for each episode

### 4. Modified `get_action()` Method
**Workflow:**
1. Encode current observation
2. Add current state to history (with `action=None` initially)
3. Send entire history to server via API
4. Receive action from server
5. Update the last history entry with the action taken
6. Return action for execution

**API Payload:**
```python
payload = {
    "task_name": task_name,
    "history": list(self.hist_context),  # All state-action pairs
    "step": self.step_count,
    "temperature": temperature,
    "max_tokens": max_tokens,
    "max_history_length": self.max_history_length
}
```

### 5. Added Command-Line Argument
```bash
--max-history-length <int>  # Default: 5
```

## Usage Example

```bash
# Run with default history length (5 frames)
python qwen_policy_client.py --task MineRLBasaltFindCave-v0

# Run with custom history length (10 frames)
python qwen_policy_client.py --task MineRLBasaltFindCave-v0 --max-history-length 10

# Run with shorter history for faster inference (3 frames)
python qwen_policy_client.py --task MineRLBasaltFindCave-v0 --max-history-length 3
```

## How History Works

### Step-by-Step Flow

**Step 1:**
- Observation: `obs_1`
- History: `[{image: obs_1, action: None}]`
- Server receives: 1 image
- Server returns: `action_1`
- History updated: `[{image: obs_1, action: action_1}]`

**Step 2:**
- Observation: `obs_2`
- History: `[{image: obs_1, action: action_1}, {image: obs_2, action: None}]`
- Server receives: 2 images + 1 action summary
- Server returns: `action_2`
- History updated: `[{image: obs_1, action: action_1}, {image: obs_2, action: action_2}]`

**Step 3:**
- Observation: `obs_3`
- History: `[{..., action: action_1}, {..., action: action_2}, {image: obs_3, action: None}]`
- Server receives: 3 images + 2 action summaries
- Server returns: `action_3`
- And so on...

### Automatic Limiting

If `max_history_length=5` and you reach step 10:
- Only the most recent 5 state-action pairs are kept
- Older frames are automatically removed from the deque
- Server receives: frames from steps 6-10

## Benefits

1. **Velocity Estimation**: VLM can detect if scene is changing or if agent is stuck
2. **Obstacle Detection**: If moving forward but scene unchanged → obstacle detected
3. **Action Context**: Model knows what was recently tried
4. **Progress Tracking**: Model can see if it's making progress toward goal
5. **Adaptive Behavior**: Can change strategy based on recent history

## Performance Considerations

- **Shorter History (3-5)**: Faster inference, good for real-time
- **Medium History (5-10)**: Balanced, recommended default
- **Longer History (10+)**: Slower but better temporal understanding

## Integration with Server

The client now sends requests in the new history format:
- Server automatically detects history mode
- Server uses new prompt with velocity/obstacle guidance
- Server processes multiple images through vision encoder
- Server provides action summaries in natural language

The old single-image format is still supported for backward compatibility, but the history mode provides significantly better performance for navigation tasks.
