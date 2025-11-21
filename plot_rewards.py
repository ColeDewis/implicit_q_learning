import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def load_file_data(file_path):
    """Reads data from a file and returns steps and rewards as numpy arrays."""
    steps, rewards = [], []
    with open(file_path, "r") as file:
        print("Found seed:", file_path.split("/")[-1])
        for line in file:
            step, reward = map(float, line.split())
            steps.append(step)
            rewards.append(reward)
    return np.array(steps), np.array(rewards)


def aggregate_data_across_seeds(folder_path):
    """Aggregates data from all files in the folder."""
    all_steps = None
    all_rewards = []

    file_paths = glob.glob(os.path.join(folder_path, "*.txt"))
    for file_path in file_paths:
        steps, rewards = load_file_data(file_path)
        if all_steps is None:
            all_steps = steps

        all_rewards.append(rewards)

    return all_steps, np.array(all_rewards)


def main():
    # Hardcoded folder path
    IQL_FOLDER_PATH = "/home/coled/655/implicit_q_learning/results/IQL/Ant_maze_hardest_noisy_multistart"
    steps, rewards = aggregate_data_across_seeds(IQL_FOLDER_PATH)

    mean_rewards = np.mean(rewards, axis=0)
    min_rewards = np.min(rewards, axis=0)
    max_rewards = np.max(rewards, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(steps, mean_rewards, label="IQL", color="blue")
    plt.fill_between(
        steps, min_rewards, max_rewards, color="blue", alpha=0.2, label="Min/Max Range"
    )

    DDQN_FOLDER_PATH = IQL_FOLDER_PATH.replace("IQL", "DDQN")
    # DDQN_FOLDER_PATH = None
    if DDQN_FOLDER_PATH is not None:
        _, ddqn_rewards = aggregate_data_across_seeds(DDQN_FOLDER_PATH)
        ddqn_mean_rewards = np.mean(ddqn_rewards, axis=0)
        ddqn_min_rewards = np.min(ddqn_rewards, axis=0)
        ddqn_max_rewards = np.max(ddqn_rewards, axis=0)
        plt.plot(steps, ddqn_mean_rewards, label="DDQN A/C", color="orange")
        plt.fill_between(
            steps,
            ddqn_min_rewards,
            ddqn_max_rewards,
            color="orange",
            alpha=0.2,
            label="Min/Max Range",
        )

    plt.xlabel("Steps")
    plt.ylabel("Reward")
    plt.title(f"Reward vs Steps on {IQL_FOLDER_PATH.split('/')[-1]}")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
