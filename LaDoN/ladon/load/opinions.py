import pickle as pkl
import glob
from re import sub
from turtle import color
from matplotlib.pyplot import legend, title, xlabel, ylabel
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import scale

sns.set(rc={"figure.figsize": (11.7, 8.27)})
sns.set_context(
    "paper",
    rc={
        "figure.figsize": (11.7, 8.27),
        "font.size": 17,
        "axes.titlesize": 50,
        "axes.labelsize": 22,
    },
    font_scale=2,
)
sns.set_style("whitegrid")
blue_pallette = sns.dark_palette("#69d", reverse=True, as_cmap=True)

list_of_simulations = glob.glob("../analysis/data/simulations/opinions/*")


def make_one_data_frame(path: str):
    with open(path, "rb") as f:
        data = pkl.load(f)
    return data


def combine_data():
    return pd.concat(
        [make_one_data_frame(path) for path in list_of_simulations], ignore_index=True
    )


df = combine_data()


def plot_trajectory_over_time(
    df,
    threshold: float,
    positive_learning_rate: float,
    negative_learning_rate: float,
    tie_dissolution: float,
    run: int = None,
):
    df = df[
        (df["threshold"] == threshold)
        & (df["positive_learning_rate"] == positive_learning_rate)
        & (df["negative_learning_rate"] == negative_learning_rate)
        & (df["tie_dissolution"] == tie_dissolution)
        # & (df["randomness"] == randomness)
    ]
    if isinstance(run, int):
        df = df[df["run"] == run]

    plotting = sns.relplot(
        data=df,
        x="time",
        y="opinions",
        units="agent",
        estimator=None,
        hue="randomness",
        alpha=0.25,
        kind="line",
        palette=blue_pallette,
        height=8,
        aspect=1.5,
    ).set(xlabel=r"$t$", ylabel=r"$O$")
    plotting._legend.remove()
    plt.legend(title=r"$R$", bbox_to_anchor=(1.15, 0.65), facecolor="white")


def plot_distribution_over_time(
    df,
    threshold: float,
    positive_learning_rate: float,
    negative_learning_rate: float,
    tie_dissolution: float,
    run: int = None,
):
    df = df[
        (df["threshold"] == threshold)
        & (df["positive_learning_rate"] == positive_learning_rate)
        & (df["negative_learning_rate"] == negative_learning_rate)
        & (df["tie_dissolution"] == tie_dissolution)
    ]
    if isinstance(run, int):
        df = df[df["run"] == run]

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # Initialize the FacetGrid object
    pal = sns.cubehelix_palette(20, rot=-0.25, light=0.7)
    g = sns.FacetGrid(
        df,
        row="time",
        hue="randomness",
        # hue="time",
        aspect=15,
        height=0.65,
        # palette=pal,
    )

    # Draw the densities in a few steps
    g.map(
        sns.kdeplot,
        "opinions",
        bw_adjust=0.5,
        clip_on=False,
        fill=True,
        alpha=0.5,
        linewidth=0.5,
    )
    g.map(sns.kdeplot, "opinions", clip_on=False, color="w", lw=2, bw_adjust=0.5)

    # passing color=None to refline() uses the hue mapping
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(
            0,
            0.2,
            label,
            fontweight="bold",
            # color=color,
            ha="left",
            va="center",
            transform=ax.transAxes,
        )

    g.map(label, "time")

    # Set the subplots to overlap
    g.figure.subplots_adjust(hspace=-0.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="", xlabel="Opinion")
    g.despine(bottom=True, left=True)


plot_distribution_over_time(
    df,
    threshold=0.8,
    positive_learning_rate=0.15,
    negative_learning_rate=0.1,
    tie_dissolution=1,
    run=6,
)

plt.savefig("../plots/Distribution_Over_Time.png")

sns.set_context(
    "paper",
    rc={
        "figure.figsize": (11.7, 8.27),
        "font.size": 17,
        "axes.titlesize": 50,
        "axes.labelsize": 25,
    },
    font_scale=2.5,
)

plot_trajectory_over_time(
    df,
    threshold=0.8,
    positive_learning_rate=0.15,
    negative_learning_rate=0.1,
    tie_dissolution=1,
    run=6,
)

plt.savefig(
    "../plots/example/Lineplot_Over_Time.png",
    dpi=300,
    bbox_inches="tight",
)
