import pickle as pkl
import glob
from re import sub
from turtle import color
from matplotlib.pyplot import title, xlabel, ylabel
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(rc={"figure.figsize": (11.7, 8.27)})
sns.set_context("talk")
blue_pallette = sns.dark_palette("#69d", reverse=True, as_cmap=True)

list_of_simulations = glob.glob("analysis/data/simulations/opinions/*")


def make_one_data_frame(path: str):
    with open(path, "rb") as f:
        data = pkl.load(f)
    return data


def combine_data():
    return pd.concat(
        [make_one_data_frame(path) for path in list_of_simulations], ignore_index=True
    )


data = combine_data()
