import pickle as pkl
import glob
from re import sub
from turtle import color
from matplotlib.pyplot import title, xlabel, ylabel
import matplotlib.pyplot as plt
from numpy import negative
import pandas as pd
import seaborn as sns
import ptitprince as pt
from ladon.helpers.helpers import rename_plot
import itertools

sns.set(rc={"figure.figsize": (11.7, 8.27)}, font_scale=1.5)
# Set the font to be serif, rather than sans
# sns.set_context("talk")
sns.set_context(
    "paper",
    rc={
        "figure.figsize": (11.7, 8.27),
        "font.size": 16,
        "axes.titlesize": 50,
        "axes.labelsize": 40,
    },
    font_scale=5,
)
sns.set_style("whitegrid")
blue_pallette = sns.dark_palette("#69d", reverse=True, as_cmap=True)

list_of_simulations = glob.glob("../analysis/data/simulations/final_state/*")


def make_one_data_frame(path: str):
    with open(path, "rb") as f:
        data = pkl.load(f)
    return pd.DataFrame(data)


def combine_data():
    return pd.concat(
        [make_one_data_frame(path) for path in list_of_simulations], ignore_index=True
    )


data = combine_data()

data["absolute_opinions"] = data["opinions"].apply(lambda x: abs(x))
data["opinion_shift"] = abs(data["initial_opinions"] - (data["opinions"]))
data["radicalization"] = data["absolute_opinions"] - abs(data["initial_opinions"])

betas = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
thresholds = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
combinations = list(itertools.product(thresholds, betas))

plotting = sns.displot(
    data=data,
    x="radicalization",
    hue="tie_dissolution",
    stat="percent",
    col="negative_learning_rate",
    row="threshold",
    common_norm=False,
    binwidth=0.05,
    kde=True,
    height=8.27,
    aspect=11.7 / 8.27,
).set(xlabel=r"$|O_F| - |O_I|$")

rename_plot(
    g=plotting,
    titles=[rf"$T = {x[0]}, \beta = {x[1]}$" for x in combinations],
    legend=r"$P(D)$",
)
# plotting.set(xlim=(-1, 1))
# plotting.figure.tick_params(labelsize=20)
# plt.tight_layout()
plotting.savefig("../plots/overall/Radicalization.png")
plotting.savefig("../plots/overall/Radicalization.pdf")

plt.clf()

correlations = (
    data.groupby(
        [
            "threshold",
            "positive_learning_rate",
            "negative_learning_rate",
            "tie_dissolution",
            "randomness",
        ]
    )["initial_opinions", "opinions"]
    .corr()
    .iloc[0::2, -1]
    .reset_index()
)
sns.set_context(
    "paper",
    rc={
        "figure.figsize": (11.7, 8.27),
        "font.size": 5,
        "axes.titlesize": 25,
        "axes.labelsize": 25,
    },
    font_scale=2,
)


def make_negative_learning_levels(negative_learning_rate: float):
    if negative_learning_rate < 0.05:
        return "Low"
    if negative_learning_rate > 0.05 and negative_learning_rate < 0.2:
        return "Medium"
    if negative_learning_rate > 0.15:
        return "High"


correlations["Negative Learning Level"] = correlations["negative_learning_rate"].apply(
    lambda x: make_negative_learning_levels(x)
)

g = sns.catplot(
    data=correlations,
    x="threshold",
    y="opinions",
    kind="box",
    col="Negative Learning Level",
    height=7,
    # hue="threshold",
    palette=sns.cubehelix_palette(8, rot=-0.25, light=0.9),
)

g.map(sns.stripplot, "threshold", "opinions", color="black", alpha=0.55, size=4)

g.set(ylabel=r"$\rho_{O_I, O_F}$", xlabel="Threshold")
for ax, titles in zip(
    g.axes.flatten(),
    [r"$\beta < 0.10$", r"$0.10 \leq \beta \leq 0.15$", r"$0.20 \leq \beta$"],
):
    ax.set_title(titles)
    ax.set_xlabel(r"$T$")

g.savefig("../plots/overall/Correlation_Initial_Opinions.png")
g.savefig("../plots/overall/Correlation_Initial_Opinions.pdf")

plt.clf()

sns.boxplot(
    data=correlations,
    x="negative_learning_rate",
    y="opinions",
    # hue="threshold",
    palette=sns.cubehelix_palette(8, rot=-0.25, light=0.9),
).set(
    ylabel=r"$\rho_{O_I, O_F}$",
    xlabel=r"$\beta$",
)
sns.stripplot(
    data=correlations,
    x="negative_learning_rate",
    y="opinions",
    color="black",
    alpha=0.4,
    size=3,
    # palette=sns.cubehelix_palette(8, rot=-0.25, light=0.9),
).set(
    ylabel=r"$\rho_{O_I, O_F}$",
    xlabel=r"$\beta$",
)
plt.savefig("../plots/overall/Correlation_Initial_Opinions_Negative_Learning_Rate.png")
plt.savefig("../plots/overall/Correlation_Initial_Opinions_Negative_Learning_Rate.pdf")

plt.clf()

sns.boxplot(
    data=correlations,
    x="threshold",
    y="opinions",
    # hue="threshold",
    palette=sns.cubehelix_palette(8, rot=-0.25, light=0.9),
).set(
    ylabel=r"$\rho_{O_I, O_F}$",
    xlabel="Threshold",
)
sns.stripplot(
    data=correlations,
    x="threshold",
    y="opinions",
    color="black",
    alpha=0.4,
    size=3
    # palette=sns.cubehelix_palette(8, rot=-0.25, light=0.9),
).set(
    ylabel=r"$\rho_{O_I, O_F}$",
    xlabel="Threshold",
)

plt.savefig("../plots/overall/Correlation_Initial_Opinions_Threshold.png")
plt.savefig("../plots/overall/Correlation_Initial_Opinions_Threshold.pdf")

plt.clf()
