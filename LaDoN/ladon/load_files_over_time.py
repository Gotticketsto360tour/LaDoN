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

sns.set(rc={"figure.figsize": (11.7, 8.27)}, font_scale=1.5)
# Set the font to be serif, rather than sans
# sns.set_context("talk")
sns.set_context(
    "paper",
    rc={
        "figure.figsize": (11.7, 8.27),
        "font.size": 13,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
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


def rename_plot(g, titles, legend):
    for ax, title in zip(g.axes.flatten(), titles):
        ax.set_title(title)
    g.legend.set(title=legend)
    g.legend.set_frame_on(True)
    return g.figure


data = combine_data()

g = sns.lineplot(
    data=data,
    x="timestep",
    y="mean_absolute_opinion",
    hue="threshold",
    palette=blue_pallette,
).set(ylabel=r"$|O|$", xlabel=r"$t$")

plt.legend(title=r"$Threshold$", bbox_to_anchor=(1.0, 0.75))
plt.savefig(
    "plots/overall/Absolute_Opinion_Threshold.png", dpi=300, bbox_inches="tight"
)

g = sns.lineplot(
    data=data,
    x="timestep",
    y="mean_absolute_opinion",
    hue="positive_learning_rate",
    palette=blue_pallette,
).set(ylabel=r"$|O|$", xlabel=r"$t$")

plt.legend(title=r"$\alpha$", bbox_to_anchor=(1.0, 0.75))
plt.savefig(
    "plots/overall/Absolute_Opinion_Positive_Learning_Rate.png",
    dpi=300,
    bbox_inches="tight",
)

g = sns.relplot(
    data=data,
    x="timestep",
    y="mean_absolute_opinion",
    kind="line",
    hue="negative_learning_rate",
    palette=blue_pallette,
).set(ylabel=r"$|O|$", xlabel=r"$t$")

plt.legend(title=r"$\beta$", bbox_to_anchor=(1.0, 0.75))
plt.savefig(
    "plots/overall/Absolute_Opinion_Negative_Learning_Rate.png",
    dpi=300,
    bbox_inches="tight",
)

g = sns.relplot(
    data=data,
    x="timestep",
    y="mean_absolute_opinion",
    hue="tie_dissolution",
    kind="line",
    col="randomness",
    # palette=blue_pallette,
).set(ylabel=r"$|O|$", xlabel=r"$t$")

rename_plot(g, titles=[r"$R = 0.1$", r"$R = 0.3$", r"$R = 0.5$"], legend=r"$P(D)$")


g = sns.lineplot(
    data=data,
    x="timestep",
    y="mean_absolute_opinion",
    hue="tie_dissolution",
    palette=blue_pallette,
).set(ylabel=r"$|O|$", xlabel=r"$t$")

plt.legend(title=r"$P(D)$", bbox_to_anchor=(1.0, 0.75))
plt.savefig(
    "plots/overall/Absolute_Opinion_Tie_Dissolution.png", dpi=300, bbox_inches="tight"
)

g = sns.lineplot(
    data=data,
    x="timestep",
    y="average_path_length",
    hue="negative_learning_rate",
    palette=blue_pallette,
).set(ylabel=r"$APL$", xlabel=r"$t$")

plt.legend(title=r"$\beta$", bbox_to_anchor=(1.0, 0.75))

g = sns.lineplot(
    data=data,
    x="timestep",
    y="average_path_length",
    hue="threshold",
    palette=blue_pallette,
).set(ylabel=r"$APL$", xlabel=r"$t$")

plt.legend(title=r"$Threshold$", bbox_to_anchor=(1.0, 0.75))

g = sns.relplot(
    data=data,
    x="timestep",
    y="mean_distance",
    hue="tie_dissolution",
    kind="line",
    col="randomness",
    palette=blue_pallette,
).set(ylabel=r"$Mean Distance$", xlabel=r"$t$")

rename_plot(g, titles=[r"$R = 0.1$", r"$R = 0.3$", r"$R = 0.5$"], legend=r"$P(D)$")

plt.legend(title=r"$P(D)$", bbox_to_anchor=(1.0, 0.75))

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

sns.boxplot(
    data=correlations,
    x="tie_dissolution",
    y="average_path_length",
    dodge=True,
    palette=sns.cubehelix_palette(8, rot=-0.25, light=0.9),
    linewidth=3,
).set(ylabel=r"$\rho_{|O|, APL}$", xlabel=r"$P(D)$")
sns.stripplot(
    data=correlations,
    x="tie_dissolution",
    y="average_path_length",
    dodge=True,
    color="black",
    alpha=0.4,
    size=3
    # palette=sns.cubehelix_palette(8, rot=-0.25, light=0.9),
).set(ylabel=r"$\rho_{|O|, APL}$", xlabel=r"$P(D)$")
plt.savefig("plots/overall/Tie_Dissolution_Correlations_Boxplot_Full.png")

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
]

g = sns.lineplot(
    data=data_specific,
    x="timestep",
    y="average_path_length",
    palette=blue_pallette,
).set(ylabel=r"$APL$", xlabel=r"$t$")
plt.savefig("plots/example/Example_Average_Path_Length.png")

g = sns.lineplot(
    data=data_specific,
    x="timestep",
    y="mean_absolute_opinion",
    palette=blue_pallette,
).set(ylabel=r"$|O|$", xlabel=r"$t$")
plt.savefig("plots/example/Example_Absolute_Opinion.png")

plt.legend(title=r"$Threshold$", bbox_to_anchor=(1.0, 0.75))
plt.show(g)

data = data.query("negative_learning_rate > 0")
data = data.query("tie_dissolution > 0")

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

g = sns.lineplot(
    data=data_merged,
    x="timestep",
    y="mean_absolute_opinion",
    hue="Final State",
    # palette=blue_pallette,
).set(ylabel=r"$|O|$", xlabel=r"$t$")

g = sns.relplot(
    data=data_merged,
    x="timestep",
    y="mean_absolute_opinion",
    col="Final State",
    # hue="Final State",
    alpha=0.1,
    kind="line",
    units="run",
    estimator=None,
    # linewidth = 0.1
    # palette=blue_pallette,
).set(ylabel=r"$|O|$", xlabel=r"$t$")

g = sns.relplot(
    data=data_merged,
    x="timestep",
    y="mean_absolute_opinion",
    col="Final State",
    # hue="polarized",
    alpha=0.1,
    kind="line",
    units="unique_condition",
    estimator=None,
    # linewidth = 0.1
    # palette=blue_pallette,
).set(ylabel=r"$|O|$", xlabel=r"$t$")

data_merged

sns.lineplot(
    data=data_merged,
    x="timestep",
    y="mean_absolute_opinion",
    hue="polarized",
    hue_order=["Polarized", "Inbetween", "Consensus"],
    estimator=None,
    alpha=0.05,
    ci=None,
    linewidth=3,
    err_style=None,
    units="unique_condition",
    palette=sns.color_palette("deep", n_colors=3),
).set(ylabel=r"$|O|$", xlabel=r"$t$")
plt.legend(
    title="Final State",
    bbox_to_anchor=(1.0, 0.6),
)

sns.lineplot(
    data=data_merged,
    x="timestep",
    y="average_path_length",
    hue="polarized",
    hue_order=["Polarized", "Inbetween", "Consensus"],
    estimator=None,
    alpha=0.5,
    ci=None,
    # linewidth=0.1,
    err_style=None,
    palette=sns.color_palette("deep", n_colors=3),
).set(ylabel=r"$APL$", xlabel=r"$t$")
plt.legend(
    title="Final State",
    bbox_to_anchor=(1.0, 0.5),
)

sns.boxplot(data=data_polarized, x="polarized", y="final_polarization")
