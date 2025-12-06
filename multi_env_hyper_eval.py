import glob
import re
from collections import defaultdict

import numpy as np


def cdf_normalize(score: float, performance_pool: list[float]) -> float:
    """
    https://arxiv.org/pdf/2407.18840
    Calculates the Cross-environment Hyperparameter Setting (CHS) CDF score.
    The score is the percentage of all performance values in the pool that are
    strictly less than the given score.

    Args:
        score: score to normalize
        performance_pool: A list of all scores recorded for the env
    """
    count_less_than_x = sum(1 for g in performance_pool if g < score)
    total_scores = len(performance_pool)

    cdf_score = count_less_than_x / total_scores

    return cdf_score


def parse_filename(filename: str):
    """
    Parses a filename like 'seed1-env=halfcheetah-hypers=lr0.1_disc0.99.txt'.
    """
    # seed number (\d+), environment name (env=.+?), and hyperparameter key (hypers=.+?)
    match = re.search(r"(?:AC_AM)?seed(\d+)-env=(.+?)-hypers=(.+?)\.txt", filename)
    if match is None:
        return None

    seed_num = int(match.group(1))
    env_name = match.group(2)
    hyper_key = match.group(3)

    return env_name, hyper_key, seed_num


def load_file_score(filepath: str):
    # TODO not sure what the format will be.
    with open(filepath, "r") as f:
        return float(f.readline().strip().split(" ")[1])


if __name__ == "__main__":
    PATH = "./hyp_res/AM_CEM/"
    files = glob.glob(PATH + "*.txt")

    # Group scores by environment and hyperparameter settings
    env_hyper_scores = defaultdict(lambda: defaultdict(list))
    num_files = 0
    for file in files:
        parse_file = parse_filename(file)
        if parse_file is None:
            continue
        num_files += 1
        env_name, hyper_key, seed_num = parse_file
        score = load_file_score(file)
        env_hyper_scores[env_name][hyper_key].append(score)

    print(f"PROCESSED {num_files} FILES")

    normalized_scores = {}
    for env_name, hyper_scores in env_hyper_scores.items():
        mean_scores = {}
        for hyper_key, scores in hyper_scores.items():
            mean_score = np.mean(scores)
            mean_scores[hyper_key] = mean_score
        print(env_name, max(mean_scores.values()))
        # Normalize scores for this environment
        pooled_results = list(mean_scores.values())
        normalized_scores[env_name] = {
            hyper_key: cdf_normalize(score, pooled_results)
            for hyper_key, score in mean_scores.items()
        }

    print("Total Normalized Hyperparameter Setting Scores:")
    total_cdf_scores = defaultdict(list)
    for env_name, hyper_scores in normalized_scores.items():
        for hyper_key, norm_score in hyper_scores.items():
            total_cdf_scores[hyper_key].append(norm_score)

    mean_cdf_scores = {
        hyper_key: np.mean(scores) for hyper_key, scores in total_cdf_scores.items()
    }
    sorted_mean_cdf_scores = dict(
        sorted(mean_cdf_scores.items(), key=lambda item: item[1], reverse=True)
    )
    for hyper_key, mean_cdf in sorted_mean_cdf_scores.items():
        print(f"  {hyper_key}: Mean Normalized Score = {mean_cdf:.4f}")

    print("\nResults by Environment:")
    for env_name, hyper_scores in normalized_scores.items():
        print(f"Environment: {env_name}")
        sorted_hyper_scores = dict(
            sorted(hyper_scores.items(), key=lambda item: item[1], reverse=True)
        )
        for hyper_key, norm_score in sorted_hyper_scores.items():
            print(f"  {hyper_key}: {norm_score:.4f}")
