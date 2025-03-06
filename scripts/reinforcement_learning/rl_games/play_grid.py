# play_grid.py
import os
import csv
import re
import subprocess

# Base directory containing all experiment folders
BASE_DIR = "/media/storage1/Sathvik/IsaacLab/logs/rl_games/double_pendulum"
# CSV output
CSV_PATH = "evaluation_results_acrobot.csv"

def main():
    results = []
    results.append(["Model Directory", "Time to Goal (s)"])

    # Loop over subdirectories that match your experiment naming pattern
    for exp_dir in sorted(os.listdir(BASE_DIR)):
        if not exp_dir.startswith("acrobot_grid") or not exp_dir.endswith("_train"):
            continue  # skip non-experiment folders
        model_dir = os.path.join(BASE_DIR, exp_dir)
        model_path = os.path.join(model_dir, "nn", "double_pendulum.pth")
        if not os.path.isfile(model_path):
            print(f"[WARNING] No checkpoint found at {model_path}, skipping.")
            results.append([exp_dir, "NO_CHECKPOINT"])
            continue

        # Prepare to call play.py
        cmd = [
            "python", "scripts/reinforcement_learning/rl_games/play.py",
            "--task", "Isaac-Double-Pendulum-Direct-v0",  # adapt to your environment ID if needed
            "--num_envs", "1",
            # "--headless",
            "--run_name", exp_dir,
            "--checkpoint", model_path
        ]
        print(f"\n[INFO] Evaluating {exp_dir} with checkpoint at {model_path}")
        # Run play.py as a subprocess, capturing its output
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
            # Check the 'TIME_TO_GOAL=' line from stdout

            match = re.search(r"TIME_TO_GOAL=([A-Za-z0-9.\-_]+)", completed.stdout)
            if match:
                time_str = match.group(1)
            else:
                # If no match, might be "TIME_TO_GOAL=FAIL" or an unhandled error
                # Check if we have "TIME_TO_GOAL=FAIL" explicitly
                fail_match = re.search(r"TIME_TO_GOAL=FAIL", completed.stdout)
                if fail_match:
                    time_str = "FAIL"
                else:
                    # No recognized pattern -> treat as error
                    time_str = f"ERROR: see logs"
            # Append to results
            results.append([exp_dir, time_str])
        except Exception as e:
            print(f"[ERROR] Subprocess play.py failed for {exp_dir}: {e}")
            results.append([exp_dir, f"ERROR: {e}"])

    # Write results to CSV
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(results)

    print(f"\n[INFO] Evaluation complete. Results saved to {CSV_PATH}")


if __name__ == "__main__":
    main()
