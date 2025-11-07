"""
Test script for QWEN VLM Agent on BASALT tasks.

Usage:
    python test_QwenVLM.py --task FindCave
    python test_QwenVLM.py --task MakeWaterfall
    python test_QwenVLM.py --task CreateVillageAnimalPen
    python test_QwenVLM.py --task BuildVillageHouse
"""

from argparse import ArgumentParser
from run_Qwen_agent import main as run_qwen_agent
from config import EVAL_EPISODES, EVAL_MAX_STEPS

# Task name mapping
TASK_MAPPING = {
    'FindCave': 'MineRLBasaltFindCave-v0',
    'MakeWaterfall': 'MineRLBasaltMakeWaterfall-v0',
    'CreateVillageAnimalPen': 'MineRLBasaltCreateVillageAnimalPen-v0',
    'BuildVillageHouse': 'MineRLBasaltBuildVillageHouse-v0',
}

def main():
    parser = ArgumentParser("Test QWEN VLM Agent on BASALT tasks")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_MAPPING.keys()),
        help="BASALT task to run"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=EVAL_EPISODES,
        help="Number of episodes to run"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=EVAL_MAX_STEPS,
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
    
    args = parser.parse_args()
    
    env_name = TASK_MAPPING[args.task]
    
    print(f"\nRunning QWEN VLM Agent on {args.task}")
    print(f"Environment: {env_name}")
    print(f"Episodes: {args.episodes}")
    print(f"Max steps: {args.max_steps}")
    print(f"Device: {args.device}\n")
    
    run_qwen_agent(
        env=env_name,
        task_name=env_name,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        show=args.show,
        device=args.device
    )

if __name__ == "__main__":
    main()
