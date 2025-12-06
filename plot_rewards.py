import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def load_file_data(file_path):
    """Reads data from a file and returns steps and rewards as numpy arrays."""
    steps, rewards = [], []
    with open(file_path, "r") as file:
        print(file_path.split("/")[-1])
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
    for paths in folder_path:
        alg_rewards = []
        file_paths = glob.glob(os.path.join(paths, "*.txt"))
        for file_path in file_paths:
            steps, rewards = load_file_data(file_path)
            if all_steps is None:
                all_steps = steps

            alg_rewards.append(rewards)
        all_rewards.append(alg_rewards)



    return all_steps, np.array(all_rewards)


def main():
    # Hardcoded folder path
    # IQL_FOLDER_PATH = "/home/coled/655/implicit_q_learning/results/IQL/Ant_maze_hardest_noisy_multistart"
    IQL_FOLDER_PATH = (
        "/home/jaspe/CMPUT 655/results (1)/hyper_sweep/halfcheetah-medium-expert-v2_halfcheetah_medium_expert-v2-AC/",
        "/home/jaspe/CMPUT 655/results (1)/hyper_sweep/halfcheetah-medium-expert-v2_halfcheetah_medium_expert-v2-CEM/",
        "/home/jaspe/CMPUT 655/results (1)/hyper_sweep/halfcheetah-medium-replay-v2_halfcheetah_medium_replay-v2-AC/",
        "/home/jaspe/CMPUT 655/results (1)/hyper_sweep/halfcheetah-medium-replay-v2_halfcheetah_medium_replay-v2-CEM/",
        "/home/jaspe/CMPUT 655/results (1)/hyper_sweep/halfcheetah-medium-v2_halfcheetah_medium-v2-AC/",
        "/home/jaspe/CMPUT 655/results (1)/hyper_sweep/halfcheetah-medium-v2_halfcheetah_medium-v2-CEM/"
        # "/home/jaspe/CMPUT 655/implicit_q_learning/DDQN_results/results/hyper_sweep/antmaze-large-play-v0_Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse/IQL",
        # "/home/jaspe/CMPUT 655/implicit_q_learning/DDQN_results/results/hyper_sweep/antmaze-large-play-v0_Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse/AC_AM",
        # "/home/jaspe/CMPUT 655/implicit_q_learning/DDQN_results/results/hyper_sweep/antmaze-large-play-v0_Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse/GA_AM",
        # "/home/jaspe/CMPUT 655/implicit_q_learning/DDQN_results/results/hyper_sweep/antmaze-large-play-v0_Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse/CEM_AM_10_10_5",
        # "/home/jaspe/CMPUT 655/implicit_q_learning/DDQN_results/results/hyper_sweep/antmaze-large-play-v0_Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse/CEM_AM_10_20_10",
        # "/home/jaspe/CMPUT 655/implicit_q_learning/DDQN_results/results/hyper_sweep/antmaze-large-play-v0_Ant_maze_hardest-maze_noisy_multistart_True_multigoal_False_sparse/CEM_AM_10_30_15",
    )

    COLS = (
        "blue",
        "red",
        "yellow",
        "orange",
        "green",
        "magenta",
    )

    LABELS = (
        "halfcheetah-medium-expert-AC",
        "halfcheetah-medium-expert-CEM",
        "halfcheetah-medium-replay-AC",
        "halfcheetah-medium-replay-CEM",
        "halfcheetah-medium-AC",
        "halfcheetah-medium-CEM",
    )
    steps, rewards = aggregate_data_across_seeds(IQL_FOLDER_PATH)

    mean_rewards = np.mean(rewards, axis=1)
    min_rewards = np.min(rewards, axis=1)
    max_rewards = np.max(rewards, axis=1)

    final_iql_mean = mean_rewards[:, -1]
    print("Final Mean Rewards:", final_iql_mean)
    plt.rcParams.update({'font.size': 18})

    plt.figure(figsize=(10, 6))
    for i, alg_mean_rewards in enumerate(mean_rewards):
        plt.plot(steps, alg_mean_rewards, label=LABELS[i], color=COLS[i])
        plt.fill_between(
            steps, min_rewards[i], max_rewards[i], color=COLS[i], alpha=0.2
        )

    # DDQN_FOLDER_PATH = IQL_FOLDER_PATH.replace("IQL", "DDQN")
    # DDQN_FOLDER_PATH = None
    # if DDQN_FOLDER_PATH is not None:
    #     _, ddqn_rewards = aggregate_data_across_seeds(DDQN_FOLDER_PATH)
    #     ddqn_mean_rewards = np.mean(ddqn_rewards, axis=0)
    #     ddqn_min_rewards = np.min(ddqn_rewards, axis=0)
    #     ddqn_max_rewards = np.max(ddqn_rewards, axis=0)
    #     final_ddqn_mean = ddqn_mean_rewards[-1]
    #     print("Final DDQN Mean Reward:", final_ddqn_mean)
    #     plt.plot(steps, ddqn_mean_rewards, label="DDQN A/C", color="orange")
    #     plt.fill_between(
    #         steps,
    #         ddqn_min_rewards,
    #         ddqn_max_rewards,
    #         color="orange",
    #         alpha=0.2,
    #         label="Min/Max Range",
    #     )

    plt.xlabel("Gradient Steps")
    plt.ylabel("Episodic Return")
    plt.title(f"antmaze-large-play-v0")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
