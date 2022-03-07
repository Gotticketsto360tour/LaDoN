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
    hue="tie_dissolution",
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

# NOTE: Randomness makes situations
# more extreme. Stable conditions
# become more stable (check variance),
# and unstable conditions even more
# unstable. This happens because
# more distant areas of the network
# becomes connected. In stable
# conditions, this allows
# more positive influence,
# which on average draws opinions towards
# the consensus.
# However, in unstable conditions,
# connecting to distant areas will
# often result in connecting to agents
# outside of the agent's threshold.
# This will lead both agent's to further radicalization.

# To really test this however,
# I guess I need to see how radicalization
# correlates with variance of opinions
# for different conditions

correlations = (
    data.groupby(
        [
            "threshold",
            "positive_learning_rate",
            "negative_learning_rate",
            "tie_dissolution",
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

sns.boxplot(
    data=correlations,
    x="threshold",
    y="opinions",
    # hue="threshold",
    palette=sns.cubehelix_palette(10, rot=-0.25, light=1.2),
).set(
    ylabel=r"$\rho_{O_I, O_F}$",
    xlabel="Threshold",
    title=r"Correlation between $O_I$ and $O_F$",
)
sns.stripplot(
    data=correlations,
    x="threshold",
    y="opinions",
    color="black",
    alpha=0.6,
    # palette=sns.cubehelix_palette(8, rot=-0.25, light=0.9),
).set(
    ylabel=r"$\rho_{O_I, O_F}$",
    xlabel="Threshold",
    title=r"Correlation between $O_I$ and $O_F$",
)
plt.savefig("plots/Correlation_Initial_Opinions_PLR.png")


correlations_centrality = (
    data.groupby(
        [
            "threshold",
            "positive_learning_rate",
            "negative_learning_rate",
            "tie_dissolution",
        ]
    )["centrality", "absolute_opinions"]
    .corr()
    .iloc[0::2, -1]
    .reset_index()
)

sns.relplot(
    data=correlations_centrality,
    x="negative_learning_rate",
    y="absolute_opinions",
    hue="threshold",
    height=8.27,
    aspect=11.7 / 8.27,
    palette=blue_pallette,
    kind="line",
).set(
    ylabel=r"$\rho_{centrality, |O_F|}$",
    xlabel="Negative Learning Rate",
    title=r"Correlation between $|O_F|$ and Centrality",
)
plt.savefig("plots/Correlation_Absolute_Centrality.png")

correlations_centrality = (
    data.groupby(
        [
            "threshold",
            "positive_learning_rate",
            "negative_learning_rate",
            "tie_dissolution",
        ]
    )["degrees", "centrality"]
    .corr()
    .iloc[0::2, -1]
    .reset_index()
)

sns.boxplot(
    data=correlations_centrality,
    x="tie_dissolution",
    y="centrality",
    # hue="threshold",
    palette=sns.cubehelix_palette(10, rot=-0.25, light=1.2),
).set(
    ylabel=r"$\rho_{Degree, Centrality}$",
    xlabel=r"$P(D)$",
    title=r"Correlation between Degree and Centrality",
)
sns.stripplot(
    data=correlations_centrality,
    x="tie_dissolution",
    y="centrality",
    color="black",
    alpha=0.6,
    # palette=sns.cubehelix_palette(8, rot=-0.25, light=0.9),
).set(
    ylabel=r"$\rho_{Degree, Centrality}$",
    xlabel=r"$P(D)$",
    title=r"Correlation between Degree and Centrality",
)

sns.boxplot(
    data=correlations_centrality,
    x="threshold",
    y="centrality",
    # hue="threshold",
    palette=sns.cubehelix_palette(10, rot=-0.25, light=1.2),
).set(
    ylabel=r"$\rho_{Degree, Centrality}$",
    xlabel=r"$Threshold$",
    title=r"Correlation between Degree and Centrality",
)
sns.stripplot(
    data=correlations_centrality,
    x="threshold",
    y="centrality",
    color="black",
    alpha=0.6,
    # palette=sns.cubehelix_palette(8, rot=-0.25, light=0.9),
).set(
    ylabel=r"$\rho_{Degree, Centrality}$",
    xlabel=r"$Threshold$",
    title=r"Correlation between Degree and Centrality",
)

ax = pt.RainCloud(
    x="threshold",
    y="centrality",
    data=correlations_centrality,
    palette=sns.cubehelix_palette(10, rot=-0.25, light=1.2),
    bw=0.2,
    width_viol=1,
    alpha=1,
    dodge=True,
)

sns.relplot(
    data=correlations_centrality,
    x="negative_learning_rate",
    y="centrality",
    hue="threshold",
    height=8.27,
    aspect=11.7 / 8.27,
    palette=blue_pallette,
    kind="line",
)
