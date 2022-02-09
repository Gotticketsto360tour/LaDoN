import pickle as pkl
import glob
import pandas as pd
import seaborn as sns

sns.set(rc={"figure.figsize": (11.7, 8.27)})
sns.set_context("talk")

list_of_simulations = glob.glob("analysis/data/simulations/*")


def make_one_data_frame(path: str):
    with open(path, "rb") as f:
        data = pkl.load(f)
    return pd.DataFrame(data)


def combine_data():
    return pd.concat(
        [make_one_data_frame(path) for path in list_of_simulations], ignore_index=True
    )


data = combine_data()

data_omit = data.dropna()

plotting = sns.histplot(
    data=data_omit,
    x="distances",
    stat="density",
    hue="threshold",
    binwidth=0.05,
    kde=True,
    common_norm=False,
).set(title="Average distance to neighbor's opinion", xlabel="Average distance")

plotting = sns.histplot(
    data=data,
    x="opinions",
    hue="threshold",
    stat="percent",
    common_norm=False,
    binwidth=0.05,
    kde=True,
).set(title="Opinion Distribution", xlabel="Opinion")

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
).set(xlabel="Opinion")

# plotting.legend.get_title().set_fontsize(30)
# Legend texts
# for text in plotting.legend.texts:
#    text.set_fontsize(30)

plotting = sns.displot(
    data=data,
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

plotting = sns.histplot(
    data=data,
    x="opinions",
    hue="negative_learning_rate",
    stat="percent",
    binwidth=0.05,
    kde=True,
    common_norm=False,
).set(title="Opinion Distribution", xlabel="Opinion")
plotting = sns.histplot(
    data=data,
    x="opinions",
    hue="positive_learning_rate",
    stat="percent",
    binwidth=0.05,
    kde=True,
    common_norm=False,
).set(title="Opinion Distribution", xlabel="Opinion")

data["absolute_opinions"] = data["opinions"].apply(lambda x: abs(x))

absolute_opinions = (
    data.groupby(
        ["threshold", "positive_learning_rate", "negative_learning_rate", "randomness"]
    )
    .agg(median_opinion=("absolute_opinions", "median"))
    .reset_index()
)

correlations = (
    data.groupby(
        ["threshold", "positive_learning_rate", "negative_learning_rate", "randomness"]
    )["initial_opinions", "opinions"]
    .corr()
    .iloc[0::2, -1]
    .reset_index()
)

distances = (
    data.groupby(
        ["threshold", "positive_learning_rate", "negative_learning_rate", "randomness"]
    )
    .agg(median_distance=("distances", "median"))
    .reset_index()
)

sns.lmplot(
    data=distances,
    x="negative_learning_rate",
    y="median_distance",
    hue="threshold",
    palette="mako",
)

sns.lmplot(
    data=correlations,
    x="positive_learning_rate",
    y="opinions",
    hue="threshold",
    palette="mako",
)

sns.lmplot(data=absolute_opinions, x="randomness", y="median_opinion", hue="threshold")
sns.lmplot(
    data=absolute_opinions,
    x="negative_learning_rate",
    y="median_opinion",
    hue="threshold",
)
