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

sns.lineplot(data=data, x="timestep", y="mean_absolute_opinion")

data["absolute_opinions"] = data["opinions"].apply(lambda x: abs(x))
data["opinion_shift"] = abs(data["initial_opinions"] - (data["opinions"]))
data["radicalization"] = data["absolute_opinions"] - abs(data["initial_opinions"])

data_omit = data.dropna().reset_index(drop=True)

# plotting = sns.displot(
#     data=data_omit,
#     x="radicalization",
#     hue="positive_learning_rate",
#     col="threshold",
#     row="negative_learning_rate",
#     stat="percent",
#     common_norm=False,
#     binwidth=0.05,
#     kde=True,
#     facet_kws=dict(margin_titles=True),
# ).set(xlabel="Opinion")

plotting = sns.displot(
    data=data_omit,
    x="radicalization",
    hue="negative_learning_rate",
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

data_omit.groupby(
    ["threshold", "positive_learning_rate", "negative_learning_rate", "randomness"]
).agg(n=("opinions", "count")).reset_index()

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
    data=data_omit,
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
    hue="threshold",
    stat="percent",
    common_norm=False,
    binwidth=0.05,
    kde=True,
).set(title="Opinion Distribution", xlabel=r"$O_F$")
plt.savefig("plots/Opinion_Distribution_Threshold.png")

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

plotting = sns.histplot(
    data=data_omit,
    x="absolute_opinions",
    hue="randomness",
    stat="percent",
    common_norm=False,
    binwidth=0.05,
    kde=True,
).set(title="Randomness leads to more extreme cases", xlabel=r"$|O_F|$")
plt.savefig("plots/Absolute_Opinion_Distribution_Randomness.png")


plotting = sns.displot(
    data=data_omit,
    x="absolute_opinions",
    hue="randomness",
    stat="percent",
    col="threshold",
    row="negative_learning_rate",
    common_norm=False,
    binwidth=0.05,
    kde=True,
).set(xlabel=r"$|O_F|$")


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

randomness_effect = (
    data_omit.groupby(
        ["threshold", "positive_learning_rate", "negative_learning_rate", "randomness"]
    )
    .agg(variance=("opinions", "var"), radicalization=("radicalization", "mean"))
    .reset_index()
)

kurtosis_data = (
    data_omit.groupby(
        ["threshold", "positive_learning_rate", "negative_learning_rate", "randomness"]
    )
    .apply(pd.DataFrame.kurtosis)[
        [
            "opinions",
            "initial_opinions",
            "degrees",
            "distances",
            "centrality",
            "absolute_opinions",
            "opinion_shift",
            "radicalization",
        ]
    ]
    .reset_index()
)

sns.relplot(
    data=kurtosis_data,
    x="threshold",
    y="opinions",
    hue="negative_learning_rate",
    kind="line",
)


plotting = sns.displot(
    data=data_omit,
    x="opinions",
    hue="negative_learning_rate",
    col="threshold",
    row="randomness",
    stat="percent",
    common_norm=False,
    binwidth=0.05,
    kde=True,
    facet_kws=dict(margin_titles=True),
).set(xlabel="Opinion")

polarization = (
    data_omit.groupby(["threshold", "positive_learning_rate", "negative_learning_rate"])
    .agg(
        median_opinions=("absolute_opinions", "median"),
        variance_opinions=("opinions", "var"),
    )
    .reset_index()
)

polarized_simulations = polarization.query("median_opinions > 0.53")

polarized_data = polarized_simulations.merge(
    data_omit,
    on=["threshold", "positive_learning_rate", "negative_learning_rate"],
    how="inner",
)

polarized_data.query("degrees < 10 & centrality > 0.01")

sns.relplot(
    data=polarization,
    x="negative_learning_rate",
    y="variance_opinions",
    hue="threshold",
    palette=blue_pallette,
    height=8.27,
    aspect=11.7 / 8.27,
    kind="line",
)

sns.relplot(
    data=polarization,
    x="positive_learning_rate",
    y="median_opinions",
    hue="threshold",
    palette=blue_pallette,
    height=8.27,
    aspect=11.7 / 8.27,
    kind="line",
)

sns.relplot(
    data=polarization,
    x="negative_learning_rate",
    y="median_opinions",
    hue="threshold",
    palette=blue_pallette,
    height=8.27,
    aspect=11.7 / 8.27,
    kind="line",
)

absolute_opinions = (
    data_omit.groupby(
        ["threshold", "positive_learning_rate", "negative_learning_rate", "randomness"]
    )
    .agg(median_opinion=("absolute_opinions", "median"))
    .reset_index()
)

correlations = (
    data_omit.groupby(
        ["threshold", "positive_learning_rate", "negative_learning_rate", "randomness"]
    )["initial_opinions", "opinions"]
    .corr()
    .iloc[0::2, -1]
    .reset_index()
)

distances = (
    data_omit.groupby(
        ["threshold", "positive_learning_rate", "negative_learning_rate", "randomness"]
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

sns.relplot(
    data=correlations,
    x="positive_learning_rate",
    y="opinions",
    hue="threshold",
    palette=blue_pallette,
    height=8.27,
    aspect=11.7 / 8.27,
    kind="line",
).set(
    ylabel=r"$\rho_{O_I, O_F}$",
    xlabel="Positive Learning Rate",
    title=r"Correlation between $O_I$ and $O_F$",
)
plt.savefig("plots/Correlation_Initial_Opinions_PLR.png")

sns.relplot(
    data=absolute_opinions,
    x="negative_learning_rate",
    y="median_opinion",
    hue="threshold",
    height=8.27,
    aspect=11.7 / 8.27,
    palette=blue_pallette,
    kind="line",
).set(
    ylabel=r"Median of $|O_F|$",
    xlabel="Negative Learning Rate",
    title="Threshold and Negative learning polarizes populations",
)
plt.savefig("plots/Median_Threshold_NLR.png")


sns.relplot(
    data=absolute_opinions,
    x="threshold",
    y="median_opinion",
    hue="randomness",
    height=8.27,
    aspect=11.7 / 8.27,
    palette=blue_pallette,
    kind="line",
)

sns.relplot(
    data=absolute_opinions,
    x="negative_learning_rate",
    y="median_opinion",
    hue="randomness",
    height=8.27,
    aspect=11.7 / 8.27,
    palette=blue_pallette,
    kind="line",
)


correlations_centrality = (
    data_omit.groupby(
        ["threshold", "positive_learning_rate", "negative_learning_rate", "randomness"]
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
    data_omit.groupby(
        ["threshold", "positive_learning_rate", "negative_learning_rate", "randomness"]
    )["degrees", "absolute_opinions"]
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
    ylabel=r"$\rho_{degrees, |O_F|}$",
    xlabel="Negative Learning Rate",
    title=r"Correlation between $|O_F|$ and Degrees",
)
plt.savefig("plots/Correlation_Absolute_Degrees.png")


correlations_centrality = (
    data_omit.groupby(
        ["threshold", "positive_learning_rate", "negative_learning_rate", "randomness"]
    )["agent_number", "absolute_opinions"]
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
)

correlations_centrality = (
    data_omit.groupby(
        ["threshold", "positive_learning_rate", "negative_learning_rate", "randomness"]
    )["degrees", "centrality"]
    .corr()
    .iloc[0::2, -1]
    .reset_index()
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

correlations_centrality = (
    data_omit.groupby(
        ["threshold", "positive_learning_rate", "negative_learning_rate", "randomness"]
    )["degrees", "agent_number"]
    .corr()
    .iloc[0::2, -1]
    .reset_index()
)

sns.relplot(
    data=correlations_centrality,
    x="randomness",
    y="agent_number",
    hue="threshold",
    height=8.27,
    aspect=11.7 / 8.27,
    palette=blue_pallette,
    kind="line",
)
