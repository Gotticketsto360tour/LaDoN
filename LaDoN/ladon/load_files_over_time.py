import itertools
import pickle as pkl
import glob
from re import sub
from turtle import color
from matplotlib.pyplot import title, xlabel, ylabel
import matplotlib.pyplot as plt
from numpy import size
import pandas as pd
import seaborn as sns
import matplotlib.patches as patches
from helpers import rename_plot

sns.set(rc={"figure.figsize": (11.7, 8.27)}, font_scale=1.5)
# Set the font to be serif, rather than sans
# sns.set_context("talk")
sns.set_style("whitegrid")
sns.set_context(
    "paper",
    rc={
        "figure.figsize": (11.7, 8.27),
        "font.size": 13,
        "axes.titlesize": 17,
        "axes.labelsize": 20,
    },
    font_scale=1.7,
)

blue_pallette = sns.dark_palette("#69d", reverse=True, as_cmap=True)

list_of_simulations = glob.glob("analysis/data/simulations/over_time/*")

# make fraction of alpha / beta and plot the effect


def make_one_data_frame(path: str):
    with open(path, "rb") as f:
        data = pkl.load(f)
    return pd.DataFrame(data)


def combine_data():
    return pd.concat(
        [make_one_data_frame(path) for path in list_of_simulations], ignore_index=True
    )


data = combine_data()

sns.lineplot(
    data=data,
    x="timestep",
    y="average_clustering",
    hue="tie_dissolution",
    palette=blue_pallette,
).set(ylabel=r"$\overline{C}$", xlabel=r"$t$", xlim=(0, 10000), ylim=(0, None))

plt.legend(title=r"$P(D)$", bbox_to_anchor=(1.15, 0.65))

sns.lineplot(
    data=data,
    x="timestep",
    y="mean_distance",
    hue="tie_dissolution",
    palette=blue_pallette,
).set(ylabel=r"$d_{O}$", xlabel=r"$t$", xlim=(0, 10000), ylim=(0, None))

plt.legend(title=r"$P(D)$", bbox_to_anchor=(1.15, 0.65))
plt.savefig(
    "plots/overall/Distance_Tie_Deletion.png",
    dpi=300,
    bbox_inches="tight",
)

sns.lineplot(
    data=data,
    x="timestep",
    y="sd_absolute_opinion",
    hue="tie_dissolution",
    palette=blue_pallette,
).set(ylabel=r"$SD_{|O|}$", xlabel=r"$t$", xlim=(0, 10000), ylim=(0, None))

plt.legend(title=r"$P(D)$", bbox_to_anchor=(1.15, 0.65))
plt.savefig(
    "plots/overall/Standard_Deviation_Absolute_Opinion_Tie_Deletion.png",
    dpi=300,
    bbox_inches="tight",
)

g = sns.lineplot(
    data=data.query("negative_learning_rate == 0"),
    x="timestep",
    y="mean_absolute_opinion",
    hue="tie_dissolution",
    palette=blue_pallette,
).set(ylabel=r"$|O|$", xlabel=r"$t$", xlim=(0, 10000), ylim=(0, None))

plt.legend(title=r"$P(D)$", bbox_to_anchor=(1.15, 0.65))
plt.savefig(
    "plots/overall/Absolute_Opinion_Tie_Deletion_Without_Negative.png",
    dpi=300,
    bbox_inches="tight",
)

# sns.set_context(
#     "paper",
#     rc={
#         "figure.figsize": (11.7, 8.27),
#         "font.size": 13,
#         "axes.titlesize": 17,
#         "axes.labelsize": 20,
#     },
#     font_scale=1.7,
# )

g = sns.lineplot(
    data=data,
    x="timestep",
    y="mean_absolute_opinion",
    hue="threshold",
    legend="full",
    palette=blue_pallette,
).set(ylabel=r"$|O|$", xlabel=r"$t$", xlim=(0, 10000))

plt.legend(title=r"$T$", bbox_to_anchor=(1.0, 0.65))
plt.savefig(
    "plots/overall/Absolute_Opinion_Threshold.png", dpi=300, bbox_inches="tight"
)

g = sns.lineplot(
    data=data,
    x="timestep",
    y="mean_absolute_opinion",
    hue="positive_learning_rate",
    palette=blue_pallette,
).set(ylabel=r"$|O|$", xlabel=r"$t$", xlim=(0, 10000))

plt.legend(title=r"$\alpha$", bbox_to_anchor=(1.0, 0.65))
plt.savefig(
    "plots/overall/Absolute_Opinion_Positive_Learning_Rate.png",
    dpi=300,
    bbox_inches="tight",
)

g = sns.lineplot(
    data=data,
    x="timestep",
    y="mean_absolute_opinion",
    hue="negative_learning_rate",
    palette=blue_pallette,
).set(ylabel=r"$|O|$", xlabel=r"$t$", xlim=(0, 10000), ylim=(0, None))

plt.legend(title=r"$\beta$", bbox_to_anchor=(1.0, 0.65))
plt.savefig(
    "plots/overall/Absolute_Opinion_Negative_Learning_Rate.png",
    dpi=300,
    bbox_inches="tight",
)

sns.set_context(
    "paper",
    rc={
        "figure.figsize": (11.7, 8.27),
        "font.size": 10,
        "axes.titlesize": 17,
        "axes.labelsize": 20,
    },
    font_scale=1.7,
)

g = sns.relplot(
    data=data,
    x="timestep",
    y="average_clustering",
    hue="tie_dissolution",
    kind="line",
    col="randomness",
    palette=blue_pallette,
).set(ylabel=r"$\overline{C}$", xlabel=r"$t$")

rename_plot(g, titles=[r"$R = 0.1$", r"$R = 0.3$", r"$R = 0.5$"], legend=r"$P(D)$")
g.savefig(
    "plots/overall/Average_Clustering_Coefficient_Ties_Deleted.png",
    dpi=300,
    bbox_inches="tight",
)

g = sns.relplot(
    data=data,
    x="timestep",
    y="average_path_length",
    hue="tie_dissolution",
    kind="line",
    col="randomness",
    palette=blue_pallette,
).set(ylabel=r"$APL*$", xlabel=r"$t$")

rename_plot(g, titles=[r"$R = 0.1$", r"$R = 0.3$", r"$R = 0.5$"], legend=r"$P(D)$")
g.savefig(
    "plots/overall/Average_Path_Length_Ties_Deleted.png", dpi=300, bbox_inches="tight"
)

g = sns.relplot(
    data=data,
    x="timestep",
    y="mean_absolute_opinion",
    hue="tie_dissolution",
    kind="line",
    col="randomness",
    palette=blue_pallette,
).set(ylabel=r"$|O|$", xlabel=r"$t$")

rename_plot(g, titles=[r"$R = 0.1$", r"$R = 0.3$", r"$R = 0.5$"], legend=r"$P(D)$")
g.savefig(
    "plots/overall/Absolute_Opinion_Tie_Dissolution.png", dpi=300, bbox_inches="tight"
)

g = sns.relplot(
    data=data,
    x="timestep",
    y="negative_ties_dissoluted",
    hue="tie_dissolution",
    kind="line",
    col="randomness",
    palette=blue_pallette,
).set(ylabel=r"$NTD$", xlabel=r"$t$")

rename_plot(g, titles=[r"$R = 0.1$", r"$R = 0.3$", r"$R = 0.5$"], legend=r"$P(D)$")
g.savefig("plots/overall/Negative_Tie_Deleted.png", dpi=300, bbox_inches="tight")


correlations = (
    data.groupby(
        [
            "tie_dissolution",
            "threshold",
            "negative_learning_rate",
            "positive_learning_rate",
            "randomness",
        ]
    )["mean_absolute_opinion", "average_path_length"]
    .corr()
    .iloc[0::2, -1]
    .reset_index()
)
sns.set_context(
    "paper",
    rc={
        "figure.figsize": (11.7, 8.27),
        "font.size": 16,
        "axes.titlesize": 22,
        "axes.labelsize": 24,
    },
    font_scale=2.5,
)

g = sns.catplot(
    data=correlations,
    x="tie_dissolution",
    y="average_path_length",
    kind="box",
    col="randomness",
    height=7,
    # hue="threshold",
    palette=sns.cubehelix_palette(8, rot=-0.25, light=0.9),
)

g.map(
    sns.stripplot,
    "tie_dissolution",
    "average_path_length",
    color="black",
    alpha=0.33,
    size=4,
)

g.set(ylabel=r"$\rho_{APL, |O|}$", xlabel=r"$P(D)$")
for ax, titles in zip(
    g.axes.flatten(),
    [r"$R = 0.1$", r"$R = 0.3$", r"$R = 0.5$"],
):
    ax.set_title(titles)
    ax.set_xlabel(r"$P(D)$")


g.savefig(
    "plots/overall/Correlation_Average_Path_Length_Absolute_Opinions.png", dpi=300
)

g = sns.boxplot(
    data=correlations,
    x="tie_dissolution",
    y="average_path_length",
    hue="randomness",
    dodge=True,
    palette=sns.cubehelix_palette(5, rot=-0.25, light=0.9),
    linewidth=3,
)  # .set(ylabel=r"$\rho_{|O|, APL}$", xlabel=r"$P(D)$")
sns.stripplot(
    data=correlations,
    x="tie_dissolution",
    y="average_path_length",
    hue="randomness",
    dodge=True,
    color="black",
    alpha=0.4,
    size=3,
    # palette=sns.cubehelix_palette(8, rot=-0.25, light=0.9),
).set(ylabel=r"$\rho_{|O|, APL}$", xlabel=r"$P(D)$")

# .set(ylabel=r"$\rho_{|O|, APL}$", xlabel=r"$P(D)$")

handles, labels = g.get_legend_handles_labels()

# When creating the legend, only use the first two elements
# to effectively remove the last two.
l = plt.legend(handles[0:3], labels[0:3], bbox_to_anchor=(1.01, 0.6), title=r"$R$")
plt.savefig(
    "plots/overall/Tie_Dissolution_Correlations_Boxplot_Full.png",
    dpi=300,
    bbox_inches="tight",
)

sns.boxplot(
    data=correlations.query("tie_dissolution > 0"),
    x="tie_dissolution",
    y="average_path_length",
    dodge=True,
    palette=sns.cubehelix_palette(8, rot=-0.25, light=0.9),
    linewidth=3,
).set(ylabel=r"$\rho_{|O|, APL}$", xlabel=r"$P(D)$")
sns.stripplot(
    data=correlations.query("tie_dissolution > 0"),
    x="tie_dissolution",
    y="average_path_length",
    dodge=True,
    color="black",
    alpha=0.4,
    size=3
    # palette=sns.cubehelix_palette(8, rot=-0.25, light=0.9),
).set(ylabel=r"$\rho_{|O|, APL}$", xlabel=r"$P(D)$")
plt.savefig("plots/overall/Tie_Dissolution_Correlations_Boxplot_Over_Zero.png")

data_specific = data[
    (data["threshold"] == 0.8)
    & (data["positive_learning_rate"] == 0.15)
    & (data["negative_learning_rate"] == 0.1)
    & (data["tie_dissolution"] == 1)
    & (data["run"] == 6)
    & (data["randomness"] == 0.1)
]

data_specific_random = data[
    (data["threshold"] == 0.8)
    & (data["positive_learning_rate"] == 0.15)
    & (data["negative_learning_rate"] == 0.1)
    & (data["tie_dissolution"] == 1)
    & (data["run"] == 6)
]

sns.set_context(
    "paper",
    rc={
        "figure.figsize": (11.7, 8.27),
        "font.size": 13,
        "axes.titlesize": 17,
        "axes.labelsize": 18,
    },
    font_scale=1.7,
)

g = sns.lineplot(
    data=data_specific_random,
    x="timestep",
    y="mean_absolute_opinion",
    hue="randomness",
    palette=blue_pallette,
).set(ylabel=r"$|O|$", xlabel=r"$t$", xlim=(0, 10000))

plt.legend(title=r"$R$", bbox_to_anchor=(1.15, 0.6))
plt.savefig("plots/example/Example_Absolute_Opinion.png", dpi=300, bbox_inches="tight")


g = sns.lineplot(
    data=data_specific_random,
    x="timestep",
    y="average_path_length",
    hue="randomness",
    palette=blue_pallette,
).set(ylabel=r"$APL$", xlabel=r"$t$", xlim=(0, 10000))

plt.legend(title=r"$R$", bbox_to_anchor=(1.15, 0.6))
plt.savefig(
    "plots/example/Example_Average_Path_Length.png", dpi=300, bbox_inches="tight"
)

g = sns.lineplot(
    data=data_specific_random,
    x="timestep",
    y="negative_ties_dissoluted",
    hue="randomness",
    palette=blue_pallette,
).set(ylabel=r"$NTD$", xlabel=r"$t$")

plt.legend(title=r"$R$", bbox_to_anchor=(1.15, 0.6))
plt.savefig(
    "plots/example/Example_Negative_Ties_Deleted.png", dpi=300, bbox_inches="tight"
)

data_polarized = (
    data.groupby(
        [
            "threshold",
            "positive_learning_rate",
            "negative_learning_rate",
            "tie_dissolution",
            "randomness",
            "run",
        ]
    )
    .agg(final_polarization=("mean_absolute_opinion", "last"))
    .reset_index()
)


def binary_polarization(value: float):
    if value >= 0.8:
        return "Polarized"
    if value <= 0.2:
        return "Consensus"
    else:
        return "Inbetween"


def make_identifier_column(
    threshold,
    randomness,
    positive_learning_rate,
    negative_learning_rate,
    tie_dissolution,
    run,
):
    return f"T{threshold}_R{randomness}_P{positive_learning_rate}_N{negative_learning_rate}_D{tie_dissolution}_{run}"


data_polarized["Final State"] = data_polarized["final_polarization"].apply(
    lambda x: binary_polarization(x)
)

data_polarized["unique_condition"] = data_polarized.apply(
    lambda x: make_identifier_column(
        x.threshold,
        x.randomness,
        x.positive_learning_rate,
        x.negative_learning_rate,
        x.tie_dissolution,
        x.run,
    ),
    axis=1,
)

data_polarized["Final State"].value_counts()

data_merged = data.merge(data_polarized)

data_min_max = (
    data_merged.groupby(["Final State", "timestep"])
    .agg(min_y=("mean_absolute_opinion", "min"), max_y=("mean_absolute_opinion", "max"))
    .reset_index()
)

data_merged = data_merged.merge(data_min_max)

data_merged = data_merged[
    (data_merged["mean_absolute_opinion"] == data_merged["min_y"])
    | (data_merged["mean_absolute_opinion"] == data_merged["max_y"])
]

sns.set_context(
    "paper",
    rc={
        "figure.figsize": (20.7, 8.27),
        "font.size": 17,
        "axes.titlesize": 17,
        "axes.labelsize": 22,
    },
    font_scale=2.2,
)

consensus = data_min_max[data_min_max["Final State"] == "Consensus"]
inbetween = data_min_max[data_min_max["Final State"] == "Inbetween"]
polarized = data_min_max[data_min_max["Final State"] == "Polarized"]

time = consensus["timestep"].values
consensus_min, consensus_max = consensus["min_y"].values, consensus["max_y"].values
inbetween_min, inbetween_max = inbetween["min_y"].values, inbetween["max_y"].values
polarized_min, polarized_max = polarized["min_y"].values, polarized["max_y"].values

plt.figure(figsize=(20.7, 8.27))

plt.fill_between(time, polarized_min, polarized_max, alpha=0.7, label="Polarized")
plt.fill_between(time, inbetween_min, inbetween_max, alpha=0.7, label="In-between")
plt.fill_between(time, consensus_min, consensus_max, alpha=0.7, label="Consensus")
plt.legend(title="Final State", bbox_to_anchor=(1.12, 0.66))
plt.xlabel(r"$t$")
plt.ylabel(r"$|O|$")
plt.savefig("plots/overall/Point_Of_No_Return.png", dpi=300, bbox_inches="tight")

g = sns.relplot(
    data=data_merged,
    x="timestep",
    y="mean_absolute_opinion",
    # col="Final State",
    hue="Final State",
    alpha=0.8,
    kind="line",
    units="Final State",
    hue_order=["Polarized", "Inbetween", "Consensus"],
    estimator=None,
    aspect=1,
    height=10
    # linewidth = 0.1
    # palette=blue_pallette,
)
g.set(ylabel=r"$|O|$", xlabel=r"$t$")

g.savefig("plots/overall/Point_Of_No_Return.png", dpi=300, bbox_inches="tight")
