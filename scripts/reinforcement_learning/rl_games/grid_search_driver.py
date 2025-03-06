import itertools
import subprocess

def main():
    """
    Grid search driver: Launches multiple processes of train.py,
    each with a different set of reward weights.
    IsaacLab can be launched fresh each time, shutting down in-between.
    """

    # The different values you want for each of the 4 components
    WEIGHT_VALUES = [0.1, 1.0, 2.0]

    all_combos = list(itertools.product(WEIGHT_VALUES, repeat=4))
    print(f"Will run {len(all_combos)} combinations in total.")

    base_cmd = [
        "python",
        "scripts/reinforcement_learning/rl_games/train.py",   # look at the path carefully
        "--task", "Isaac-Double-Pendulum-Direct-v0",
        "--max_iterations", "1000",
        "--headless",
        "--num_envs", "4096",
        "--enable_cameras",
        "--video",
    ]

    for i, (w1, w2, w3, w4) in enumerate(all_combos):
        # Format the four weights
        weights_str = f"[{w1},{w2},{w3},{w4}]"

        # Unique run_name so each run logs separately
        run_name = f"acrobot_grid_{i:03d}_{w1}-{w2}-{w3}-{w4}"

        # We'll override e.g. DoublePendulumEnvCfg.reward_state_weights
        # OR env_cfg.reward_state_weights
        # If your environment is recognized as DoublePendulumEnvCfg, do:
        #   f"DoublePendulumEnvCfg.reward_state_weights={weights_str}"
        # If it's recognized as env_cfg, do:
        #   f"env_cfg.reward_state_weights={weights_str}"

        override_str = f"env.reward_state_weights={weights_str}"

        cmd = base_cmd + [
            "--run_name", run_name,
            # The Hydra override goes last:
            override_str,
        ]

        print("=" * 80)
        print(f"Launching grid iteration {i+1}/{len(all_combos)} with {weights_str}")
        print("Command: ", " ".join(cmd))
        print("=" * 80)

        subprocess.run(cmd, check=True)

    print("All grid runs completed!")

if __name__ == "__main__":
    main()
