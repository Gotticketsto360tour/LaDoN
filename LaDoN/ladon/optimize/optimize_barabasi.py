from statistics import mean
from ladon.helpers.optimize_helpers import run_single_simulation, run_optimization


def objective(trial, target, repeats, target_dictionary) -> float:
    N_TARGET = target.number_of_nodes()
    K = trial.suggest_int("K", 1, 20)
    dictionary = {
        "N_TARGET": N_TARGET,
        "N_TIMESTEPS": N_TARGET * 20,
        "K": K,
        "RECORD": False,
    }

    results = [
        run_single_simulation(dictionary, run, target, target_dictionary, "barabasi")
        for run in range(repeats)
    ]

    return mean(results)


if __name__ == "__main__":
    run_optimization(objective=objective, type="barabasi")
