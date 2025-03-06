import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=200, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--sigma", type=str, default=None, help="The policy's initial standard deviation.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--run_name", type=str, default=None, help="Name of the run.")

# NEW: A flag to enable the internal grid search
parser.add_argument("--grid_search", action="store_true", default=False, help="Run multiple training sessions for different reward weights.")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Always enable cameras if we want to record video
if args_cli.video:
    args_cli.enable_cameras = True

# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch Omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import math
import os
import random
from datetime import datetime

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config


def run_single_training(env_cfg, agent_cfg, args_cli, resume_path=None, train_sigma=None):
    """
    Perform one RL-Games training run given env_cfg and agent_cfg.
    This logic was originally in main(...).
    """
    import os
    import gymnasium as gym
    import math
    from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper
    from rl_games.common import env_configurations, vecenv
    from rl_games.common.algo_observer import IsaacAlgoObserver
    from rl_games.torch_runner import Runner
    from isaaclab.utils.dict import print_dict
    from isaaclab.utils.io import dump_pickle, dump_yaml
    from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
    
    
    # Set up logging directories
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)

    run_name = args_cli.run_name if args_cli.run_name else "unknown_task"
    log_dir = f"{run_name}_train"
    print(f"[INFO] Logging experiment in directory: {log_root_path}/{log_dir}")

    # RL-Games logging config
    agent_cfg["params"]["config"]["train_dir"] = log_root_path
    agent_cfg["params"]["config"]["full_experiment_name"] = log_dir

    # Dump env_cfg & agent_cfg
    os.makedirs(os.path.join(log_root_path, log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_root_path, log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_root_path, log_dir, "params", "agent.pkl"), agent_cfg)

    # RL-Games device + environment clip settings
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # Create environment
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # If multi-agent, convert to single-agent
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Video
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap environment for RL-Games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # Set number of actors
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs

    # Create RL-Games runner
    runner = Runner(IsaacAlgoObserver())
    runner.load(agent_cfg)

    # Train
    runner.reset()
    if resume_path is not None:
        runner.run({"train": True, "play": False, "sigma": train_sigma, "checkpoint": resume_path})
    else:
        runner.run({"train": True, "play": False, "sigma": train_sigma})

    env.close()
    
@hydra_task_config(args_cli.task, "rl_games_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with RL-Games agent, optionally performing an internal grid search."""

    # ------------------------------------------
    # Basic overrides from CLI
    # ------------------------------------------
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["params"]["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["params"]["seed"]

    agent_cfg["params"]["config"]["max_epochs"] = (
        args_cli.max_iterations if args_cli.max_iterations is not None 
        else agent_cfg["params"]["config"]["max_epochs"]
    )
    resume_path = None
    if args_cli.checkpoint is not None:
        resume_path = retrieve_file_path(args_cli.checkpoint)
        agent_cfg["params"]["load_checkpoint"] = True
        agent_cfg["params"]["load_path"] = resume_path
        print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    train_sigma = float(args_cli.sigma) if args_cli.sigma is not None else None

    if args_cli.distributed:
        agent_cfg["params"]["seed"] += AppLauncher(args_cli).global_rank
        agent_cfg["params"]["config"]["device"] = f"cuda:{AppLauncher(args_cli).local_rank}"
        agent_cfg["params"]["config"]["device_name"] = f"cuda:{AppLauncher(args_cli).local_rank}"
        agent_cfg["params"]["config"]["multi_gpu"] = True
        env_cfg.sim.device = f"cuda:{AppLauncher(args_cli).local_rank}"

    env_cfg.seed = agent_cfg["params"]["seed"]

    # ------------------------------------------
    # If not using grid_search, do a single run
    # ------------------------------------------
    if not args_cli.grid_search:
        # We'll rely on --run_name or the default
        run_single_training(env_cfg, agent_cfg, args_cli, resume_path, train_sigma)
        return

    # ------------------------------------------
    # GRID SEARCH MODE
    # ------------------------------------------
    # Example grid. We'll vary the first 4 entries of reward_state_weights.
    # Adjust as needed. We only do a few combos to illustrate.
    from copy import deepcopy
    import itertools

    WEIGHT_VALUES = [0.1, 1.0, 2.0]
    all_combos = list(itertools.product(WEIGHT_VALUES, repeat=4))

    print("[INFO] Starting GRID SEARCH with the following combos:")
    for combo in all_combos:
        print("   ", combo)

    # We'll do multiple runs in the same process,
    # so we'll close & re-init env each time
    for i, (w1, w2, w3, w4) in enumerate(all_combos):
        # Make a fresh copy so we don't mutate the original
        local_env_cfg = deepcopy(env_cfg)
        local_agent_cfg = deepcopy(agent_cfg)

        # Overwrite reward weights in Python
        # Make sure your environment code references local_env_cfg.reward_state_weights.
        local_env_cfg.reward_state_weights = [w1, w2, w3, w4]

        # Unique run_name for logging
        # e.g. "grid_000_0.1-0.1-1.0-2.0"
        sub_run_name = f"grid_{i:03d}_{w1}-{w2}-{w3}-{w4}"

        # We'll pass it via args_cli so that run_single_training uses it for logging
        old_run_name = args_cli.run_name
        args_cli.run_name = old_run_name + sub_run_name 

        print("=" * 80)
        print(f"Grid search iteration {i}/{len(all_combos)}. Weights = {w1,w2,w3,w4}")
        print("=" * 80)

        # Run
        run_single_training(local_env_cfg, local_agent_cfg, args_cli, resume_path, train_sigma)

        # Restore the old run_name if needed
        args_cli.run_name = old_run_name
        


if __name__ == "__main__":
    # Run the main function
    main()
    # close sim app
    simulation_app.close()
