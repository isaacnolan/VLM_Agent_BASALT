# QWEN VLM Policy Server

## Overview

FastAPI-based server that provides real-time action predictions for MineRL BASALT tasks using the Qwen2-VL model. This server architecture allows for:

- **Centralized model hosting**: Load the model once, serve multiple clients
- **Network-based inference**: Separate computation from game execution
- **Easy scaling**: Can be deployed on powerful GPU servers
- **API-based integration**: Simple REST API for any client

## Architecture

```
┌─────────────────┐
│  MineRL Client  │
│  (Game Loop)    │
└────────┬────────┘
         │ HTTP Request
         │ (Base64 Image)
         ▼
┌─────────────────┐
│  FastAPI Server │
│  Port 8001      │
├─────────────────┤
│  QWEN VLM Model │
│  (Loaded Once)  │
└────────┬────────┘
         │ Action Response
         │ (JSON + Reasoning)
         ▼
┌─────────────────┐
│  MineRL Client  │
│  (Execute Action)│
└─────────────────┘
```

## Files

### 1. `qwen_policy_server.py`
Main FastAPI server that:
- Loads QWEN VLM model on startup
- Provides `/get_action` endpoint for action predictions
- Handles image decoding and VLM inference
- Returns structured action responses with reasoning

### 2. `qwen_policy_client.py`
Client library and example that:
- Provides `QwenPolicyClient` class for easy integration
- Handles observation encoding and request formatting
- Demonstrates usage with MineRL environment
- Includes error handling and timeouts

## Installation

### Server Requirements

```bash
# Install FastAPI and server dependencies
pip install fastapi uvicorn

# Install QWEN VLM dependencies
pip install transformers>=4.37.0
pip install qwen-vl-utils
pip install torch torchvision
pip install pillow
pip install accelerate

# MineRL (optional, only needed for client)
pip install minerl aicrowd-gym
```

### GPU Requirements

- **Qwen2-VL-2B** (default): ~6GB VRAM
- **Qwen2-VL-7B**: ~16GB VRAM

## Usage

### 1. Start the Server

```bash
# Start on default port 8001
python qwen_policy_server.py

# Server will:
# - Load the model (takes ~30 seconds)
# - Listen on http://0.0.0.0:8001
# - Print "QWEN VLM Policy Server ready!" when ready
```

Expected output:
```
INFO:__main__:Loading QWEN VLM model...
INFO:__main__:Using device: cuda
INFO:__main__:Successfully loaded Qwen/Qwen2-VL-2B-Instruct
INFO:__main__:QWEN VLM Policy Server ready!
INFO:     Uvicorn running on http://0.0.0.0:8001
```

### 2. Use the Client

#### Option A: Using the Client Library

```python
from qwen_policy_client import QwenPolicyClient
import aicrowd_gym

# Initialize client
client = QwenPolicyClient(server_url="http://localhost:8001")

# Check server status
health = client.check_health()
print(health)  # {'status': 'healthy', 'model': 'Qwen2-VL-2B-Instruct', ...}

# Use in game loop
env = aicrowd_gym.make("MineRLBasaltFindCave-v0")
obs = env.reset()
client.reset()

for step in range(1000):
    # Get action from server
    action = client.get_action(obs, task_name="MineRLBasaltFindCave-v0")
    
    # Execute in environment
    obs, reward, done, info = env.step(action)
    
    if done:
        break
```

#### Option B: Direct HTTP Requests

```python
import requests
import base64
import numpy as np
from PIL import Image
import io

# Encode observation
pov = obs['pov']  # numpy array from MineRL
image = Image.fromarray(pov.astype(np.uint8))
buffered = io.BytesIO()
image.save(buffered, format="PNG")
img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

# Make request
response = requests.post(
    "http://localhost:8001/get_action",
    json={
        "task_name": "MineRLBasaltFindCave-v0",
        "image": {"data": img_base64},
        "step": 1,
        "temperature": 0.7,
        "max_tokens": 512
    }
)

result = response.json()
action = result['action']
reasoning = result['reasoning']
```

### 3. Run Example Client

```bash
# Make sure server is running first!
python qwen_policy_client.py
```

## API Endpoints

### GET `/`
Health check endpoint.

**Response:**
```json
{
    "status": "online",
    "service": "QWEN VLM Policy Server",
    "model_loaded": true,
    "device": "cuda"
}
```

### GET `/health`
Detailed health check.

**Response:**
```json
{
    "status": "healthy",
    "model": "Qwen2-VL-2B-Instruct",
    "device": "cuda",
    "ready": true
}
```

### POST `/get_action`
Get action prediction from VLM.

**Request Body:**
```json
{
    "task_name": "MineRLBasaltFindCave-v0",
    "image": {
        "data": "base64_encoded_image_string"
    },
    "step": 1,
    "temperature": 0.7,
    "max_tokens": 512
}
```

**Response:**
```json
{
    "action": {
        "forward": 1,
        "back": 0,
        "left": 0,
        "right": 0,
        "jump": 0,
        "sneak": 0,
        "sprint": 0,
        "attack": 0,
        "use": 0,
        "drop": 0,
        "inventory": 0,
        "camera": [5.0, -10.0],
        "hotbar.1": 0,
        "hotbar.2": 0,
        ...
        "ESC": 0
    },
    "reasoning": "I see a dark opening in the cliff ahead. Moving forward to investigate if it's a cave entrance.",
    "step": 1
}
```

## Configuration

### Server Configuration

Edit `qwen_policy_server.py`:

```python
# Change model
model_name = "Qwen/Qwen2-VL-7B-Instruct"  # Use larger model

# Change port
uvicorn.run(app, host="0.0.0.0", port=8001)  # Default: 8001

# Adjust generation parameters
generated_ids = model.generate(
    **inputs,
    max_new_tokens=512,      # Longer responses
    temperature=0.7,         # More/less random (0.0-1.0)
    top_p=0.9,              # Nucleus sampling
)
```

### Client Configuration

```python
# Custom server URL
client = QwenPolicyClient(server_url="http://your-server:8001")

# Adjust request timeout
response = requests.post(..., timeout=60)  # Default: 30 seconds

# Custom temperature
action = client.get_action(obs, task_name, temperature=0.5)
```

## Performance

### Latency

| Component | Time |
|-----------|------|
| Network (local) | ~1-5ms |
| Image encoding | ~5-10ms |
| VLM inference (2B, GPU) | ~1-2 seconds |
| Response parsing | ~1-2ms |
| **Total** | **~1-2 seconds/step** |

### Throughput

- Single GPU server: ~1 request/second (2B model)
- Can serve multiple clients sequentially
- Consider request queuing for high traffic

### Optimization Tips

1. **Use smaller model**: 2B instead of 7B
2. **Reduce max_tokens**: 256 instead of 512
3. **Lower temperature**: Faster sampling
4. **Batch requests**: If serving multiple environments
5. **Use quantization**: Int8 or Int4 for faster inference

## Deployment

### Local Deployment

```bash
# Start server
python qwen_policy_server.py
```

### Docker Deployment

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install dependencies
RUN pip install fastapi uvicorn transformers qwen-vl-utils pillow accelerate

# Copy server code
COPY qwen_policy_server.py .

# Expose port
EXPOSE 8001

# Run server
CMD ["python", "qwen_policy_server.py"]
```

Build and run:
```bash
docker build -t qwen-policy-server .
docker run --gpus all -p 8001:8001 qwen-policy-server
```

### Production Deployment

For production use:

```bash
# Use gunicorn with multiple workers
gunicorn qwen_policy_server:app \
    --workers 1 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8001 \
    --timeout 120

# Or use uvicorn with auto-reload
uvicorn qwen_policy_server:app \
    --host 0.0.0.0 \
    --port 8001 \
    --reload \
    --log-level info
```

## Troubleshooting

### Server Won't Start

**Problem**: Model fails to load
```
Solution: Check GPU availability and VRAM
- Use CPU: Set device = "cpu" in load_model()
- Use smaller model: Change to Qwen2-VL-2B
```

**Problem**: Port already in use
```
Solution: Change port in server code or kill existing process
lsof -i :8001
kill -9 <PID>
```

### Client Connection Issues

**Problem**: Connection refused
```
Solution: Ensure server is running
curl http://localhost:8001/health
```

**Problem**: Request timeout
```
Solution: Increase timeout or optimize server
client.get_action(..., timeout=60)
```

### Slow Inference

**Problem**: Taking >5 seconds per request
```
Solution:
1. Check GPU utilization: nvidia-smi
2. Use smaller model (2B)
3. Reduce max_tokens
4. Check network latency if remote
```

## Comparison with Local Inference

| Aspect | Server | Local |
|--------|--------|-------|
| Setup | Load model once | Load per instance |
| Memory | Shared across clients | Per instance |
| Latency | +network overhead | Direct |
| Scaling | Easy (multiple clients) | Difficult |
| Development | Separate concerns | Simpler code |
| Debugging | Centralized logs | Distributed |

## Integration Examples

### With Evaluation Pipeline

```python
from qwen_policy_client import QwenPolicyClient
from evaluation import evaluate_model

# Run episode with policy server
client = QwenPolicyClient()
env = aicrowd_gym.make(task_name)
obs = env.reset()

# Collect final state
for step in range(max_steps):
    action = client.get_action(obs, task_name)
    obs, _, done, _ = env.step(action)
    if done:
        break

# Save final observation
final_image_path = "final_state.png"
Image.fromarray(obs['pov']).save(final_image_path)

# Evaluate
score = evaluate_model(task_name, final_image_path)
```

### With Multiple Environments

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def run_environment(env_id, client):
    env = aicrowd_gym.make(task_name)
    obs = env.reset()
    
    for step in range(1000):
        action = await asyncio.get_event_loop().run_in_executor(
            None, client.get_action, obs, task_name
        )
        obs, _, done, _ = env.step(action)
        if done:
            break
    
    env.close()

# Run 4 environments in parallel
client = QwenPolicyClient()
await asyncio.gather(*[
    run_environment(i, client) for i in range(4)
])
```

## Future Enhancements

1. **Request Queueing**: Handle multiple simultaneous requests
2. **Caching**: Cache actions for similar observations
3. **Batching**: Process multiple images in one inference
4. **Model Switching**: Dynamic model selection per request
5. **Metrics**: Track latency, throughput, error rates
6. **Authentication**: API key-based access control
7. **Rate Limiting**: Prevent abuse
8. **WebSocket**: Real-time streaming for lower latency

## License

See LICENSE file in repository.
