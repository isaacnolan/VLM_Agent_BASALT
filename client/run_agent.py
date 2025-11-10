"""
Main script for running QWEN VLM Policy Client

This script runs the agent on MineRL BASALT tasks using the QWEN policy server.
"""

import os
import sys
import traceback
import logging
import json
from argparse import ArgumentParser
from client.policy_client import QwenPolicyClient
from time import sleep
from client.episode_recorder import EpisodeRecorder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main(task_name, n_episodes=3, max_steps=100, show=False, record_dir=None, 
         server_url="http://localhost:8001", max_history_length=5):
    """
    Run QWEN policy client with optional video recording.
    
    Args:
        task_name: MineRL BASALT task name
        n_episodes: Number of episodes to run
        max_steps: Maximum steps per episode
        show: Whether to render the environment
        record_dir: Directory to save videos and episode data (None to disable)
        server_url: URL of the QWEN policy server
        max_history_length: Maximum number of state-action pairs to keep in history
    """
    import aicrowd_gym
    import minerl  # Need to import minerl to register environments
    
    # Initialize client
    client = QwenPolicyClient(server_url=server_url, max_history_length=max_history_length)
    
    # Check server health
    health = client.check_health()
    while not health:
        logger.error("Server is not available! Waiting...")
        sleep(5)
        health = client.check_health()
    
    logger.info(f"Server Status: {json.dumps(health, indent=2)}")
    
    # Prepare recording directory if requested
    if record_dir:
        os.makedirs(record_dir, exist_ok=True)
        logger.info(f"Recording to directory: {record_dir}")
    
    # Create environment
    env = aicrowd_gym.make(task_name)
    
    logger.info(f"\nRunning agent with QWEN policy server on {task_name}")
    logger.info(f"Episodes: {n_episodes}, Max steps per episode: {max_steps}")
    logger.info("="*60)
    
    for ep in range(n_episodes):
        logger.info(f"\n--- Episode {ep + 1}/{n_episodes} ---")
        
        try:
            obs = env.reset()
        except Exception:
            # Provide more context to help debug Malmo/Minecraft startup failures.
            logger.error("Error during env.reset() - dumping diagnostics:")
            traceback.print_exc()
            logger.error("Common causes: Java (OpenJDK) missing, Malmo failed to start, X11/Xvfb not available, or permissions issues.")
            raise
        
        client.reset()
        
        # Create recorder for this episode if requested
        recorder = None
        if record_dir:
            recorder = EpisodeRecorder(record_dir, ep)
            recorder.start_recording(obs)
        
        try:
            for step in range(max_steps):
                # Get action from policy server
                action = client.get_action(obs, task_name)
                
                # ESC is not part of the predictions model
                action["ESC"] = 0
                
                # Step environment
                obs, reward, done, info = env.step(action)
                
                if show:
                    env.render()
                
                # Write frame from observation if recording
                if recorder:
                    recorder.write_frame(obs, step)
                
                if done:
                    logger.info(f"Episode finished at step {step + 1}")
                    break
        
        finally:
            # Always stop recorder for this episode, even if interrupted
            if recorder:
                recorder.stop_recording()
            
            # Save episode data (actions and reasoning)
            if record_dir:
                client.save_episode_data(record_dir, ep)
    
    env.close()
    logger.info("\nDone!")


if __name__ == "__main__":
    parser = ArgumentParser("Run QWEN VLM policy client on MineRL BASALT environment")
    
    parser.add_argument("--task", type=str, default="MineRLBasaltFindCave-v0",
                        help="BASALT task name (default: MineRLBasaltFindCave-v0)")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to run (default: 3)")
    parser.add_argument("--max-steps", type=int, default=100,
                        help="Maximum steps per episode (default: 100)")
    parser.add_argument("--show", action="store_true",
                        help="Render the environment")
    parser.add_argument("--record-dir", type=str, default=None,
                        help="Directory to save episode videos and data (default: None)")
    parser.add_argument("--server-url", type=str, default="http://localhost:8001",
                        help="URL of the QWEN policy server (default: http://localhost:8001)")
    parser.add_argument("--max-history-length", type=int, default=5,
                        help="Maximum number of state-action pairs to keep in history (default: 5)")
    
    args = parser.parse_args()
    
    main(
        task_name=args.task,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        show=args.show,
        record_dir=args.record_dir,
        server_url=args.server_url,
        max_history_length=args.max_history_length
    )
