from statistics import mean
from typing import Dict
from ladon.helpers.optimize_helpers import run_single_simulation, run_optimization
import networkx as nx


def objective(
    trial: int, target: nx.Graph(), repeats: int, target_dictionary: Dict
) -> float:
    """Objective function for Optuna

    Args:
        trial (int): Trial number used by Optuna
        target (nx.Graph): Target network to match by simulation
        repeats (int): Integer specifying how many samples are used to estimate goodness of fit
        target_dictionary (Dict): Dictionary containing precomputed values

    Returns:
        float: Mean goodness of fit of the simulated network
    """

    N_TARGET = target.number_of_nodes()
    N_EDGES = target.number_of_edges()
    K = round(N_EDGES / N_TARGET)
    threshold = trial.suggest_float("threshold", 0, 2)
    randomness = trial.suggest_float("randomness", 0, 1)
    positive_learning_rate = trial.suggest_float("positive_learning_rate", 0, 0.5)
    negative_learning_rate = trial.suggest_float("negative_learning_rate", 0, 0.5)
    tie_dissolution = trial.suggest_float("tie_dissolution", 0, 1)

    dictionary = {
        "THRESHOLD": threshold,
        "N_TARGET": N_TARGET,
        "RANDOMNESS": randomness,
        "N_TIMESTEPS": N_TARGET * 10,  # 20
        "POSITIVE_LEARNING_RATE": positive_learning_rate,
        "NEGATIVE_LEARNING_RATE": negative_learning_rate,
        "P": 0.5,
        "K": 2 * K,
        "TIE_DISSOLUTION": tie_dissolution,
        "RECORD": False,
    }

    results = [
        run_single_simulation(dictionary, run, target, target_dictionary, "opinion")
        for run in range(repeats)
    ]

    return mean(results)


if __name__ == "__main__":
    run_optimization(objective=objective, type="")
