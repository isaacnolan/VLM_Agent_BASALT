# QWEN VLM Agent for MineRL BASALT

## Overview

This implementation uses the **Qwen2-VL** (Vision-Language Model) as a policy function to make decisions in MineRL BASALT tasks. The agent processes visual observations from the Minecraft environment and generates actions based on both the image input and task-specific text prompts.

## Architecture

### Key Components

1. **QwenVLMAgent Class** (`run_Qwen_agent.py`)
   - Main agent class that interfaces between MineRL and the QWEN VLM
   - Handles observation processing, prompt engineering, and action generation

2. **Observation Processing**
   - Converts MineRL's POV (Point of View) numpy arrays to PIL Images
   - Formats images for QWEN VLM input processing

3. **Prompt Engineering**
   - Task-specific system prompts and instructions
   - Step-by-step guidance for the VLM
   - Structured JSON response format for action parsing

4. **Action Generation**
   - VLM generates reasoning and actions in JSON format
   - Parser converts JSON to MineRL-compatible action dictionaries
   - Handles camera movements, button presses, and inventory actions

## Model Information

### Supported Models

- **Qwen2-VL-2B-Instruct**: ~4 Billion parameters (recommended for this implementation)
- **Qwen2-VL-7B-Instruct**: ~7 Billion parameters (more capable but slower)

The 4B parameter model provides a good balance between performance and speed for real-time Minecraft gameplay.

## Installation

### Prerequisites

```bash
# Install transformers and related packages
pip install transformers>=4.37.0
pip install qwen-vl-utils
pip install torch torchvision
pip install pillow
pip install accelerate

# MineRL and BASALT dependencies (if not already installed)
pip install minerl
pip install aicrowd-gym
```

### Model Download

The models will be automatically downloaded from Hugging Face on first run. Make sure you have sufficient disk space:
- Qwen2-VL-2B: ~8GB
- Qwen2-VL-7B: ~28GB

## Usage

### Basic Usage

```bash
# Run on FindCave task
python run_Qwen_agent.py --env MineRLBasaltFindCave-v0

# Run on MakeWaterfall task  
python run_Qwen_agent.py --env MineRLBasaltMakeWaterfall-v0

# Run with visualization
python run_Qwen_agent.py --env MineRLBasaltFindCave-v0 --show

# Run on CPU (slower)
python run_Qwen_agent.py --env MineRLBasaltFindCave-v0 --device cpu
```

### Using Test Script

```bash
# Run on FindCave task
python test_QwenVLM.py --task FindCave

# Run with custom episodes and steps
python test_QwenVLM.py --task MakeWaterfall --episodes 5 --max_steps 1000

# Run with visualization
python test_QwenVLM.py --task CreateVillageAnimalPen --show
```

### Command Line Arguments

- `--env`: MineRL BASALT environment name (required)
- `--n_episodes`: Number of episodes to run (default: 3)
- `--max_steps`: Maximum steps per episode (default: 1e9)
- `--show`: Render the environment
- `--device`: Device to run on ('cuda' or 'cpu', default: 'cuda')
- `--model`: Hugging Face model name (default: 'Qwen/Qwen2-VL-7B-Instruct')

## Task-Specific Prompts

Each BASALT task has customized prompts:

### FindCave
- **Goal**: Locate and identify cave entrances
- **Strategy**: Exploration, looking for dark openings in terrain
- **Key actions**: Movement, camera control, navigation

### MakeWaterfall
- **Goal**: Create an aesthetically pleasing waterfall
- **Strategy**: Find elevated terrain, place water sources
- **Key actions**: Block placement, water bucket usage, terrain modification

### CreateVillageAnimalPen
- **Goal**: Build a functional animal pen in a village
- **Strategy**: Locate village, gather materials, build enclosure
- **Key actions**: Fence placement, gate installation, area enclosure

### BuildVillageHouse
- **Goal**: Construct a house matching village aesthetic
- **Strategy**: Analyze village style, gather materials, build structure
- **Key actions**: Block placement, architectural design, integration

## How It Works

### Step-by-Step Process

1. **Observation Capture**
   ```python
   obs = env.reset()
   image = agent._observation_to_image(obs)  # Convert to PIL Image
   ```

2. **Prompt Construction**
   ```python
   messages = [
       {"role": "system", "content": task_config['system']},
       {"role": "user", "content": [
           {"type": "image", "image": image},
           {"type": "text", "text": prompt}
       ]}
   ]
   ```

3. **VLM Inference**
   ```python
   inputs = processor(text=[text], images=image_inputs, ...)
   generated_ids = model.generate(**inputs, max_new_tokens=512)
   response = processor.decode(generated_ids)
   ```

4. **Action Parsing**
   ```python
   action_dict = parse_json_response(response)
   minerl_action = convert_to_minerl_format(action_dict)
   ```

5. **Environment Step**
   ```python
   obs, reward, done, info = env.step(minerl_action)
   ```

## Action Space

The agent can control:

### Movement
- `forward`, `back`: Forward/backward movement
- `left`, `right`: Strafing
- `jump`: Jumping
- `sneak`: Crouching
- `sprint`: Sprinting

### Interaction
- `attack`: Break blocks
- `use`: Place blocks/use items
- `drop`: Drop items
- `inventory`: Open inventory

### Camera
- `camera`: [horizontal_angle, vertical_angle] in degrees

### Hotbar
- `hotbar.1` through `hotbar.9`: Select hotbar slot

## Expected Response Format

The VLM is expected to respond in this JSON format:

```json
{
    "reasoning": "I see a dark opening in the cliff face ahead. This appears to be a cave entrance. I should move forward and look more directly at it to confirm.",
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
        "camera": [5.0, -10.0]
    }
}
```

## Performance Considerations

### GPU Memory
- **Qwen2-VL-2B**: ~6GB VRAM
- **Qwen2-VL-7B**: ~16GB VRAM

### Inference Speed
- **GPU (RTX 3090)**: ~1-2 seconds per step (2B model)
- **CPU**: ~5-10 seconds per step (2B model)

### Tips for Better Performance
1. Use the 2B model for faster inference
2. Use GPU if available
3. Reduce `max_new_tokens` for faster generation
4. Use batch processing if running multiple parallel environments

## Customization

### Modifying Prompts

Edit the `TASK_PROMPTS` dictionary in `run_Qwen_agent.py`:

```python
TASK_PROMPTS["MineRLBasaltFindCave-v0"]["task_description"] = """
Your custom task description here...
"""
```

### Changing Model

Use a different QWEN model:

```bash
python run_Qwen_agent.py --env MineRLBasaltFindCave-v0 \
    --model Qwen/Qwen2-VL-2B-Instruct
```

### Adjusting Generation Parameters

Modify in `get_action()` method:

```python
generated_ids = self.model.generate(
    **inputs,
    max_new_tokens=512,      # Adjust response length
    temperature=0.7,          # Adjust randomness (0.0-1.0)
    top_p=0.9,               # Nucleus sampling
    do_sample=True,          # Enable sampling
)
```

## Troubleshooting

### Out of Memory
- Use smaller model (2B instead of 7B)
- Reduce batch size
- Use CPU (slower but uses system RAM)

### Slow Inference
- Use GPU
- Reduce `max_new_tokens`
- Use smaller model

### Invalid Actions
- Check JSON parsing in `_parse_vlm_response()`
- Verify prompt clarity
- Add more examples to prompts

### Model Download Errors
- Check internet connection
- Verify Hugging Face access
- Try manual download: `huggingface-cli download Qwen/Qwen2-VL-2B-Instruct`

## Comparison with VPT Agent

| Feature | QWEN VLM Agent | VPT Agent |
|---------|----------------|-----------|
| Input | Visual observations | Visual observations |
| Architecture | Vision-Language Model | CNN + Transformer |
| Decision Making | Language-based reasoning | Learned behavior patterns |
| Training | Pre-trained, zero-shot | Requires behavioral cloning |
| Flexibility | High (can follow new instructions) | Low (task-specific) |
| Speed | Slower (~1-2s/step) | Faster (~0.01s/step) |
| Interpretability | High (provides reasoning) | Low (black box) |

## Future Improvements

1. **Memory/Context**: Add conversation history for multi-step reasoning
2. **Chain-of-Thought**: Implement step-by-step reasoning prompts
3. **Few-Shot Learning**: Add example actions in prompts
4. **Fine-tuning**: Fine-tune on Minecraft-specific data
5. **Action Caching**: Cache similar observations for faster response
6. **Hierarchical Planning**: Implement high-level goal planning

## Citation

If using this implementation, please cite:

```bibtex
@software{qwen_vlm_basalt_agent,
  title={QWEN VLM Agent for MineRL BASALT},
  author={Your Name},
  year={2025},
  url={https://github.com/isaacnolan/VLM_Agent_BASALT}
}
```

## License

See LICENSE file in the repository.
