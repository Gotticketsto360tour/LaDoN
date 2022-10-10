from typing import Dict
from scipy import rand
from ladon.classes.network import Network
from ladon.helpers.optimize_helpers import make_network_by_seed
import numpy as np
from ladon.config import (
    THRESHOLDS,
    RANDOMNESS,
    POSITIVE_LEARNING_RATES,
    NEGATIVE_LEARNING_RATES,
    TIE_DISSOLUTIONS,
)
import random
import itertools
import multiprocessing as mp
import pickle as pkl
import os
import pandas as pd


def make_one_simulation(
    threshold: float,
    randomness: float,
    positive_learning_rate: float,
    negative_learning_rate: float,
    tie_dissolution: float,
) -> None:

    dictionary = {
        "THRESHOLD": threshold,
        "N_TARGET": 1000,  # 1000
        "RANDOMNESS": randomness,
        "N_TIMESTEPS": 10000,
        "POSITIVE_LEARNING_RATE": positive_learning_rate,
        "NEGATIVE_LEARNING_RATE": negative_learning_rate,
        "P": 0.5,
        "K": 7,
        "TIE_DISSOLUTION": tie_dissolution,
        "RECORD": True,
    }

    networks = [make_network_by_seed(dictionary, run) for run in range(10)]

    out_dict_final_state = {
        "threshold": threshold,
        "randomness": randomness,
        "positive_learning_rate": positive_learning_rate,
        "negative_learning_rate": negative_learning_rate,
        "tie_dissolution": tie_dissolution,
        "opinions": np.hstack(
            [network.get_opinion_distribution() for network in networks]
        ),
        "initial_opinions": np.hstack(
            [network.get_initial_opinion_distribution() for network in networks]
        ),
        "degrees": np.hstack(
            [network.get_degree_distribution() for network in networks]
        ),
        "distances": np.hstack(
            [network.get_opinion_distances() for network in networks]
        ),
        "centrality": np.hstack([network.get_centrality() for network in networks]),
        "clustering": np.hstack([network.get_clustering() for network in networks]),
    }

    out_dict_over_time = {
        "threshold": threshold,
        "randomness": randomness,
        "positive_learning_rate": positive_learning_rate,
        "negative_learning_rate": negative_learning_rate,
        "tie_dissolution": tie_dissolution,
        "mean_distance": np.hstack(
            [np.array(network.MEAN_DISTANCE) for network in networks]
        ),
        "negative_ties_dissoluted": np.hstack(
            [np.array(network.NEGATIVE_TIES_DISSOLUTED) for network in networks]
        ),
        "mean_absolute_opinion": np.hstack(
            [np.array(network.MEAN_ABSOLUTE_OPINIONS) for network in networks]
        ),
        "sd_absolute_opinion": np.hstack(
            [np.array(network.SD_ABSOLUTE_OPINIONS) for network in networks]
        ),
        "average_path_length": np.hstack(
            [np.array(network.AVERAGE_PATH_LENGTH) for network in networks]
        ),
        "average_clustering": np.hstack(
            [np.array(network.AVERAGE_CLUSTERING) for network in networks]
        ),
        "assortativity": np.hstack(
            [np.array(network.ASSORTATIVITY) for network in networks]
        ),
        "timestep": np.array(
            [timestep for network in networks for timestep in range(0, 10000, 20)]
        ),
        "run": np.array(
            [
                run
                for run, network in enumerate(networks)
                for timestep in range(0, 10000, 20)
            ]
        ),
    }

    data_list = [
        [
            {
                "threshold": threshold,
                "randomness": randomness,
                "positive_learning_rate": positive_learning_rate,
                "negative_learning_rate": negative_learning_rate,
                "tie_dissolution": tie_dissolution,
                "time": (time + 1) * 500,  # dictionary.get("N_TARGET")
                "run": run,
                "agent": [agent for agent in range(dictionary.get("N_TARGET"))],
                "opinions": opinion,
            }
            for time, opinion in enumerate(network.OPINION_DISTRIBUTIONS)
        ]
        for run, network in enumerate(networks)
    ]

    df = pd.concat(
        [
            pd.concat([pd.DataFrame(x) for x in data], ignore_index=True)
            for data in data_list
        ],
        ignore_index=True,
    )

    with open(
        f"../analysis/data/simulations/opinions/T{threshold}-R{randomness}-P{positive_learning_rate}-N{negative_learning_rate}-D{tie_dissolution}.pkl",
        "wb",
    ) as handle:
        pkl.dump(df, handle, protocol=pkl.HIGHEST_PROTOCOL)

    with open(
        f"../analysis/data/simulations/final_state/T{threshold}-R{randomness}-P{positive_learning_rate}-N{negative_learning_rate}-D{tie_dissolution}.pkl",
        "wb",
    ) as handle:
        pkl.dump(out_dict_final_state, handle, protocol=pkl.HIGHEST_PROTOCOL)

    with open(
        f"../analysis/data/simulations/over_time/T{threshold}-R{randomness}-P{positive_learning_rate}-N{negative_learning_rate}-D{tie_dissolution}.pkl",
        "wb",
    ) as handle:
        pkl.dump(out_dict_over_time, handle, protocol=pkl.HIGHEST_PROTOCOL)


def make_all_simulations() -> None:

    values = [
        THRESHOLDS,
        RANDOMNESS,
        POSITIVE_LEARNING_RATES,
        NEGATIVE_LEARNING_RATES,
        TIE_DISSOLUTIONS,
    ]

    combinations = list(itertools.product(*values))
    combinations = [
        one_combination
        for one_combination in combinations
        if not os.path.exists(
            f"../analysis/data/simulations/over_time/T{one_combination[0]}-R{one_combination[1]}-P{one_combination[2]}-N{one_combination[3]}-D{one_combination[4]}.pkl"
        )
    ]

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.starmap(make_one_simulation, combinations)


if __name__ == "__main__":
    make_all_simulations()
