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

list_of_simulations = glob.glob("analysis/data/simulations/over_time/*")


def make_one_data_frame(path: str):
    with open(path, "rb") as f:
        data = pkl.load(f)
    return pd.DataFrame(data)


def combine_data():
    return pd.concat(
        [make_one_data_frame(path) for path in list_of_simulations], ignore_index=True
    )


data = combine_data()
data_without = data.query("threshold != 1.2")
data_without = data_without.query("threshold != 0.6")
data = data_without

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

g = sns.lineplot(
    data=data,
    x="timestep",
    y="mean_absolute_opinion",
    hue="negative_learning_rate",
    palette=blue_pallette,
).set(ylabel=r"$|O|$", xlabel=r"$t$")

plt.legend(title=r"$\beta$", bbox_to_anchor=(1.0, 0.75))
plt.savefig(
    "plots/overall/Absolute_Opinion_Negative_Learning_Rate.png",
    dpi=300,
    bbox_inches="tight",
)

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

g = sns.lineplot(
    data=data,
    x="timestep",
    y="mean_distance",
    hue="tie_dissolution",
    palette=blue_pallette,
).set(ylabel=r"$Mean Distance$", xlabel=r"$t$")

plt.legend(title=r"$P(D)$", bbox_to_anchor=(1.0, 0.75))

g = sns.lineplot(
    data=data,
    x="timestep",
    y="average_clustering",
    hue="threshold",
    palette=blue_pallette,
).set(ylabel=r"$Mean Distance$", xlabel=r"$t$")

plt.legend(title=r"$P(D)$", bbox_to_anchor=(1.0, 0.75))

correlations = (
    data_without.groupby(
        [
            "tie_dissolution",
            "threshold",
            "negative_learning_rate",
            "positive_learning_rate",
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
    alpha=0.6,
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
    alpha=0.6,
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
