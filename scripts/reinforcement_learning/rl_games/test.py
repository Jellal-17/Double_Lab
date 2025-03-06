# play.py
import argparse
import math
import os
import time
import torch
import gymnasium as gym

from isaaclab.app import AppLauncher
from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner
from datetime import datetime

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true", help="Use the pre-trained checkpoint from Nucleus.")
parser.add_argument("--use_last_checkpoint", action="store_true", help="When no checkpoint provided, use the last saved model.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--run_name", type=str, default=None, help="Name of the run.")

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Always enable cameras to record video if requested
if args_cli.video:
    args_cli.enable_cameras = True

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


def main():
    """Play with RL-Games agent and measure time to goal."""
    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    # The log directory
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    formatted_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = args_cli.run_name if args_cli.run_name else "unknown_task"
    log_dir = f"{run_name}_{formatted_time}_train"
    print(f"[INFO] Logging experiment in directory: {log_root_path}/{log_dir}")

    # Resolve checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately, no published pre-trained checkpoint is available.")
            return
    elif args_cli.checkpoint is None:
        # Use the last or best checkpoint from logs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth" if not args_cli.use_last_checkpoint else ".*"
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        # Use the path given by user
        resume_path = retrieve_file_path(args_cli.checkpoint)

    # Setup environment for RL-Games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Record video if requested
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper",
        lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # Load model into RL-Games agent
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()
    if hasattr(agent, "is_deterministic"):
        agent.is_deterministic = True

    # Setup environment run
    dt = env.unwrapped.physics_dt
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]

    # For multi-batch logic
    _ = agent.get_batch_size(obs, 1)
    if agent.is_rnn:
        agent.init_rnn()

    # Extract or define the goal state
    # This example uses env.unwrapped.cfg.goal_state & success_threshold if present
    if hasattr(env.unwrapped.cfg, "goal_state") and hasattr(env.unwrapped.cfg, "success_threshold"):
        goal_state = torch.tensor(env_cfg.goal_state, dtype=torch.float32)
        # Double-pendulum angles are normalized to [-pi, pi]
        goal_state[:2] = (goal_state[:2] + math.pi) % (2 * math.pi) - math.pi
        success_threshold = env_cfg.success_threshold
    else:
        # Fallback
        goal_state = torch.tensor([math.pi, 0.0, 0.0, 0.0], dtype=torch.float32)
        success_threshold = 0.1

    max_steps = int(env.unwrapped.cfg.episode_length_s / dt) if hasattr(env.unwrapped.cfg, "episode_length_s") else 1000
    goal_reached = False
    step_count = 0

    # Loop until we reach the goal or run out of steps
    while True:
        with torch.inference_mode():
            obs_tensor = agent.obs_to_torch(obs)
            actions = agent.get_action(obs_tensor, is_deterministic=agent.is_deterministic)
        obs, _, dones, _ = env.step(actions)
        if isinstance(obs, dict):
            obs = obs["obs"]

        step_count += 1
        # Check if environment ended
        if len(dones) > 0 and any(dones):
            # Episode ended by env (time out, etc.)
            break

        # Check if goal is reached
        state = torch.tensor(obs, dtype=torch.float32)
        diff = state - goal_state
        if torch.norm(diff) < success_threshold:
            goal_reached = True
            break

        if step_count >= max_steps:
            break

    # Once done, print the time or fail
    if goal_reached:
        time_to_goal = dt * step_count
        print(f"TIME_TO_GOAL={time_to_goal:.4f}")
    else:
        print("TIME_TO_GOAL=FAIL")

    # Cleanup
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
