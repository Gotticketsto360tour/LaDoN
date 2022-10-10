from statistics import mean
from ladon.helpers.optimize_helpers import run_optimization, run_single_simulation


def objective(trial, target, repeats, target_dictionary) -> float:
    N_TARGET = target.number_of_nodes()
    N_EDGES = target.number_of_edges()
    K = round(N_EDGES / N_TARGET)
    randomness = trial.suggest_float("randomness", 0, 1)

    dictionary = {
        "N_TARGET": N_TARGET,
        "RANDOMNESS": randomness,
        "N_TIMESTEPS": N_TARGET * 10,
        "P": 0.5,
        "K": 2 * K,
        "RECORD": False,
    }

    results = [
        run_single_simulation(dictionary, run, target, target_dictionary, "no_opinion")
        for run in range(repeats)
    ]

    return mean(results)


if __name__ == "__main__":
    run_optimization(objective=objective, type="no_opinion")
