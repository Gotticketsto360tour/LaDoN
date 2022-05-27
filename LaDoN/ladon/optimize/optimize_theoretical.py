from statistics import mean
from unittest import result
from ladon.config import NAME_DICTIONARY
import networkx as nx
import numpy as np
import optuna
import netrd
from tqdm import tqdm
from ladon.helpers.optimize_helpers import run_single_simulation, run_optimization
import random
import pickle as pkl
import multiprocessing as mp
import joblib
import plotly.io as pio
import math

pio.renderers.default = "notebook"


def objective(trial, target, repeats, target_dictionary) -> float:
    N_TARGET = target.number_of_nodes()
    N_EDGES = target.number_of_edges()
    K = round(N_EDGES / N_TARGET)
    P = trial.suggest_float("P", 0, 1)

    dictionary = {
        "N_TARGET": N_TARGET,
        "N_TIMESTEPS": N_TARGET * 20,
        "P": P,
        "K": 2 * K,
        "RECORD": False,
    }

    results = [
        run_single_simulation(dictionary, run, target, target_dictionary, "theoretical")
        for run in range(repeats)
    ]

    return mean(results)


if __name__ == "__main__":
    run_optimization(objective=objective, type="theoretical")
