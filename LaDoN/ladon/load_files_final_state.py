import pickle as pkl
import glob
from re import sub
from turtle import color
from matplotlib.pyplot import title, xlabel, ylabel
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import ptitprince as pt

sns.set(rc={"figure.figsize": (11.7, 8.27)})
sns.set_context("talk")
blue_pallette = sns.dark_palette("#69d", reverse=True, as_cmap=True)

list_of_simulations = glob.glob("analysis/data/simulations/final_state/*")


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

sns.histplot(
    data=data,
    x="opinions",
    hue="threshold",
    stat="percent",
    common_norm=False,
    binwidth=0.05,
    kde=True,
).set(title="Opinion Distribution", xlabel=r"$O_F$")

sns.histplot(
    data=data,
    x="opinions",
    hue="randomness",
    stat="percent",
    common_norm=False,
    binwidth=0.05,
    kde=True,
).set(title="Opinion Distribution", xlabel=r"$O_F$")


sns.histplot(
    data=data,
    x="opinions",
    hue="tie_dissolution",
    stat="percent",
    common_norm=False,
    binwidth=0.05,
    kde=True,
).set(title="Opinion Distribution", xlabel=r"$O_F$")

plotting = sns.displot(
    data=data,
    x="radicalization",
    hue="negative_learning_rate",
    stat="percent",
    common_norm=False,
    binwidth=0.05,
    kde=True,
    height=8.27,
    aspect=11.7 / 8.27,
).set(xlabel=r"$|O_F| - |O_I|$")

plotting = sns.displot(
    data=data,
    x="radicalization",
    hue="tie_dissolution",
    stat="percent",
    common_norm=False,
    binwidth=0.05,
    kde=True,
    height=8.27,
    aspect=11.7 / 8.27,
).set(xlabel=r"$|O_F| - |O_I|$")

# facet = sns.FacetGrid(data, col="threshold", row="negative_learning_rate")
# facet.map_dataframe(
#     lambda data, color: sns.heatmap(
#         data_omit[
#             [
#                 "absolute_opinions",
#                 "opinions",
#                 "distances",
#                 "opinion_shift",
#                 "initial_opinions",
#                 "degrees",
#                 "centrality",
#                 "agent_number",
#             ]
#         ].corr()
#     )
# )

sns.heatmap(
    data[
        [
            "absolute_opinions",
            "opinions",
            "distances",
            "opinion_shift",
            "initial_opinions",
            "degrees",
            "centrality",
            "clustering",
        ]
    ].corr(),
    annot=True,
)

plotting = sns.histplot(
    data=data,
    x="opinion_shift",
    stat="percent",
    hue="negative_learning_rate",
    binwidth=0.05,
    kde=True,
    common_norm=False,
).set(title="Opinion shift", xlabel=r"$|O_I - O_F|$")

plotting = sns.histplot(
    data=data,
    x="distances",
    stat="percent",
    hue="positive_learning_rate",
    binwidth=0.05,
    kde=True,
    common_norm=False,
).set(title="Average distance to neighbor's opinion", xlabel="Average distance")

sns.histplot(
    data=data,
    x="opinions",
    hue="positive_learning_rate",
    stat="percent",
    common_norm=False,
    binwidth=0.05,
    kde=True,
).set(title="Opinion Distribution", xlabel=r"$O_F$")
plt.savefig("plots/Opinion_Distribution_PLR.png")

sns.histplot(
    data=data,
    x="opinions",
    hue="negative_learning_rate",
    stat="percent",
    common_norm=False,
    binwidth=0.05,
    kde=True,
).set(title="Opinion Distribution", xlabel=r"$O_F$")
plt.savefig("plots/Opinion_Distribution_NLR.png")


plotting = sns.displot(
    data=data,
    x="opinions",
    hue="positive_learning_rate",
    col="threshold",
    row="negative_learning_rate",
    stat="percent",
    common_norm=False,
    binwidth=0.05,
    kde=True,
    facet_kws=dict(margin_titles=True),
).set(xlabel=r"$O_F$")
plt.savefig("plots/Opinion_Distribution_Facet.png")

# plotting.legend.get_title().set_fontsize(30)
# Legend texts
# for text in plotting.legend.texts:
#    text.set_fontsize(30)

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

distances = (
    data.groupby(
        [
            "threshold",
            "positive_learning_rate",
            "negative_learning_rate",
            "tie_dissolution",
            "randomness",
        ]
    )
    .agg(median_distance=("distances", "median"))
    .reset_index()
)

sns.relplot(
    data=distances,
    x="negative_learning_rate",
    y="median_distance",
    hue="threshold",
    palette=blue_pallette,
    height=8.27,
    aspect=11.7 / 8.27,
    kind="line",
)
sns.violinplot(
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

sns.violinplot(
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

plt.savefig("plots/overall/Correlation_Initial_Opinions_Threshold.png")
