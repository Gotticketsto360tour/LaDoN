import pickle as pkl
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.io as pio
import optuna
import joblib
import networkx as nx
from ladon.helpers.helpers import find_average_path, get_main_component
from ladon.config import NAME_DICTIONARY
import plotly.io as pio
import ptitprince as pt

sns.set(rc={"figure.figsize": (11.7, 8.27)}, font_scale=1.5)
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
sns.set_style("whitegrid")

blue_pallette = sns.dark_palette("#69d", reverse=True, as_cmap=True)


def change_labels(string: str) -> str:
    if string == "Opinion_Model":
        return "Co-evolutionary model"
    elif string == "No_Opinion_Model":
        return "Network Formation model"
    elif string == "Target":
        return "Empirical network"
    elif string == "Small-world Network":
        return "Small-world network"
    elif string == "Scale-free Network":
        return "Scale-free network"


def change_network_labels(string: str) -> str:
    translation = {
        "dolphin": "Dolphins",
        "karate": "Karate Club",
        "netscience": "Citation Network",
        "polblogs": "Political Blogs",
        "polbooks": "Political Books",
        "politicians": "Politicians",
        "tvshows": "TV Shows",
    }
    return translation.get(string)


def subtract_empirical_values(df: pd.DataFrame, empirical_values: pd.DataFrame):
    df["clustering"] = df["clustering"] - empirical_values["clustering"]
    df["average_path"] = df["average_path"] - empirical_values["average_path"]
    return df


def find_difference(df: pd.DataFrame):
    list_of_subsets = []
    for network in df["network"].unique():
        subset = df[df["network"] == network]
        empirical_values = subset[subset["type"] == "Empirical network"]
        subset = subtract_empirical_values(subset, empirical_values)
        list_of_subsets.append(subset)
    return pd.concat(list_of_subsets)


with open(
    f"../analysis/data/optimization/data_from_all_runs.pkl",
    "rb",
) as handle:
    data = pkl.load(handle)

data = pd.concat([pd.DataFrame(x) for x in data])

data["type"] = data["type"].apply(lambda x: change_labels(x))
data["network"] = data["network"].apply(lambda x: change_network_labels(x))

data = find_difference(data)

data.query("type != 'Empirical network'").groupby("type").agg(
    clustering_med=("clustering", "median"),
    clustering_iqr=("clustering", lambda x: x.quantile(0.75) - x.quantile(0.25)),
    average_path_med=("average_path", "median"),
    average_path_iqr=("average_path", lambda x: x.quantile(0.75) - x.quantile(0.25)),
    JSD_path_med=("JSD_paths", "median"),
    JSD_path_iqr=("JSD_paths", lambda x: x.quantile(0.75) - x.quantile(0.25)),
    JSD_med=("JSD_degree", "median"),
    JSD_iqr=("JSD_degree", lambda x: x.quantile(0.75) - x.quantile(0.25)),
    mean_med=("mean", "median"),
    mean_iqr=("mean", lambda x: x.quantile(0.75) - x.quantile(0.25)),
).reset_index().round(3)

data_melt = data.melt(id_vars=["type", "network", "run", "average_path"])

sns.set_context(
    "paper",
    rc={
        "figure.figsize": (11.7, 8.27),
        "font.size": 13,
        "axes.titlesize": 17,
        "axes.labelsize": 14,
    },
    font_scale=1.3,
)

g = sns.FacetGrid(
    data=data_melt.query("type != 'Empirical network'"),
    col="variable",
    sharex=False,
    height=4.5,
    aspect=1,
    gridspec_kws={"wspace": 0.2},
    legend_out=True,
)

g.map(
    pt.half_violinplot,
    "value",
    "type",
    bw=0.2,
    cut=0.0,
    scale="area",
    width=0.6,
    inner=None,
    palette=sns.color_palette(n_colors=4),
)
g.map(
    sns.stripplot,
    "value",
    "type",
    edgecolor="gray",
    size=2,
    jitter=1,
    zorder=0,
    palette=sns.color_palette(n_colors=4),
)

g.set_ylabels("")
g.set_titles("")
for ax, name in zip(
    g.axes.flatten(), [r"$\overline{C}$", r"$JSD(D)$", r"$JSD(P)$", r"$O(A,G)$"]
):
    ax.set_xlabel(name)
# g.axes[0][0].set(xlim=(-0.75, 0.25))
# g.axes[0][1].set(xlim=(-3, 3))
g.axes[0][2].set(xlim=(0, 0.65))

# g.tight_layout()

g.savefig("../plots/overall/Model_Evaluation_Overview.png", dpi=300)
g.savefig("../plots/overall/Model_Evaluation_Overview.pdf", dpi=300)

plt.clf()


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

g = sns.boxplot(
    data=data.query("type != 'Empirical network'"),
    y="network",
    x="clustering",
    hue="type",
    order=[
        "Karate Club",
        "Dolphins",
        "Political Books",
        "Citation Network",
        "Political Blogs",
        "TV Shows",
        "Politicians",
    ][::-1],
    # capsize=0.07,
    # hue_order=["Co-evolutionary model", "Network Formation model"],
)
sns.stripplot(
    data=data.query("type != 'Empirical network'"),
    x="clustering",
    y="network",
    hue="type",
    dodge=True,
    order=[
        "Karate Club",
        "Dolphins",
        "Political Books",
        "Citation Network",
        "Political Blogs",
        "TV Shows",
        "Politicians",
    ][::-1],
    edgecolor="gray",
    linewidth=1.5,
    jitter=1,
    # color="black",
    alpha=0.8,
    size=5,
    # hue_order=["Co-evolutionary model", "Network Formation model"],
    # palette=sns.cubehelix_palette(8, rot=-0.25, light=0.9),
)
plt.xlabel(r"$\overline{C}$")
plt.ylabel("")
plt.axvline(x=0, color="black", ls="--")
plt.xlim(-0.8, 0.3)
g.hlines(
    [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
    xmin=-0.8,
    xmax=0.3,
    colors="gray",
    linestyles="dotted",
)
# plt.ylim(0,5)
plt.ylim(-0.5, 6.5)
handles, labels = g.get_legend_handles_labels()

# When creating the legend, only use the first two elements
# to effectively remove the last two.
l = plt.legend(
    handles[0:4], labels[0:4], title="Type of Model", bbox_to_anchor=(1.4, 0.6)
)

plt.savefig(
    "../plots/overall/Model_Evaluation_Average_Clustering.png",
    dpi=300,
    bbox_inches="tight",
)
plt.savefig(
    "../plots/overall/Model_Evaluation_Average_Clustering.pdf",
    dpi=300,
    bbox_inches="tight",
)
plt.clf()


g = sns.boxplot(
    data=data.query("type != 'Empirical network'"),
    y="network",
    x="average_path",
    hue="type",
    order=[
        "Karate Club",
        "Dolphins",
        "Political Books",
        "Citation Network",
        "Political Blogs",
        "TV Shows",
        "Politicians",
    ][::-1],
    # capsize=0.07,
    # hue_order=["Co-evolutionary model", "Network Formation model"],
)
sns.stripplot(
    data=data.query("type != 'Empirical network'"),
    x="average_path",
    y="network",
    hue="type",
    dodge=True,
    order=[
        "Karate Club",
        "Dolphins",
        "Political Books",
        "Citation Network",
        "Political Blogs",
        "TV Shows",
        "Politicians",
    ][::-1],
    edgecolor="gray",
    linewidth=1.5,
    jitter=1,
    # color="black",
    alpha=0.8,
    size=5,
    # hue_order=["Co-evolutionary model", "Network Formation model"],
    # palette=sns.cubehelix_palette(8, rot=-0.25, light=0.9),
)
plt.xlabel(r"$APL*$")
plt.ylabel("")
plt.axvline(x=0, color="black", ls="--")
plt.xlim(-3, 3.5)
g.hlines(
    [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
    xmin=-3,
    xmax=3.5,
    colors="gray",
    linestyles="dotted",
)
# plt.ylim(0,5)
plt.ylim(-0.5, 6.5)
handles, labels = g.get_legend_handles_labels()

# When creating the legend, only use the first two elements
# to effectively remove the last two.
l = plt.legend(
    handles[0:4], labels[0:4], title="Type of Model", bbox_to_anchor=(1.4, 0.6)
)

plt.savefig("../plots/overall/Model_Evaluation_APL.png", dpi=300, bbox_inches="tight")
plt.savefig("../plots/overall/Model_Evaluation_APL.pdf", dpi=300, bbox_inches="tight")
plt.clf()

g = sns.boxplot(
    data=data.query("type != 'Empirical network'"),
    y="network",
    x="JSD_degree",
    hue="type",
    order=[
        "Karate Club",
        "Dolphins",
        "Political Books",
        "Citation Network",
        "Political Blogs",
        "TV Shows",
        "Politicians",
    ][::-1],
    # capsize=0.07,
    # hue_order=["Co-evolutionary model", "Network Formation model"],
)
sns.stripplot(
    data=data.query("type != 'Empirical network'"),
    x="JSD_degree",
    y="network",
    hue="type",
    dodge=True,
    order=[
        "Karate Club",
        "Dolphins",
        "Political Books",
        "Citation Network",
        "Political Blogs",
        "TV Shows",
        "Politicians",
    ][::-1],
    edgecolor="gray",
    linewidth=1.5,
    jitter=1,
    # color="black",
    alpha=0.8,
    size=5,
    # hue_order=["Co-evolutionary model", "Network Formation model"],
    # palette=sns.cubehelix_palette(8, rot=-0.25, light=0.9),
)
plt.xlabel(r"$JSD(D)$")
plt.ylabel("")
plt.xlim(0, 0.8)
g.hlines(
    [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
    xmin=0,
    xmax=0.8,
    colors="gray",
    linestyles="dotted",
)
# plt.axvline(x=0, color = "black", ls="--")
# plt.ylim(0,5)
plt.ylim(-0.5, 6.5)
handles, labels = g.get_legend_handles_labels()

# When creating the legend, only use the first two elements
# to effectively remove the last two.
l = plt.legend(
    handles[0:4], labels[0:4], title="Type of Model", bbox_to_anchor=(1.4, 0.6)
)

plt.savefig("../plots/overall/Model_Evaluation_JSD.png", dpi=300, bbox_inches="tight")
plt.savefig("../plots/overall/Model_Evaluation_JSD.pdf", dpi=300, bbox_inches="tight")

plt.clf()


g = sns.boxplot(
    data=data.query("type != 'Empirical network'"),
    y="network",
    x="JSD_paths",
    hue="type",
    order=[
        "Karate Club",
        "Dolphins",
        "Political Books",
        "Citation Network",
        "Political Blogs",
        "TV Shows",
        "Politicians",
    ][::-1],
    # capsize=0.07,
    # hue_order=["Co-evolutionary model", "Network Formation model"],
)
sns.stripplot(
    data=data.query("type != 'Empirical network'"),
    x="JSD_paths",
    y="network",
    hue="type",
    dodge=True,
    order=[
        "Karate Club",
        "Dolphins",
        "Political Books",
        "Citation Network",
        "Political Blogs",
        "TV Shows",
        "Politicians",
    ][::-1],
    edgecolor="gray",
    linewidth=1.5,
    jitter=1,
    # color="black",
    alpha=0.8,
    size=5,
    # hue_order=["Co-evolutionary model", "Network Formation model"],
    # palette=sns.cubehelix_palette(8, rot=-0.25, light=0.9),
)
plt.xlabel(r"$JSD(P)$")
plt.ylabel("")
plt.xlim(0, 0.8)
g.hlines(
    [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
    xmin=0,
    xmax=0.8,
    colors="gray",
    linestyles="dotted",
)
# plt.axvline(x=0, color = "black", ls="--")
# plt.ylim(0,5)
plt.ylim(-0.5, 6.5)
handles, labels = g.get_legend_handles_labels()

# When creating the legend, only use the first two elements
# to effectively remove the last two.
l = plt.legend(
    handles[0:4], labels[0:4], title="Type of Model", bbox_to_anchor=(1.4, 0.6)
)

g = sns.boxplot(
    data=data.query("type != 'Empirical network'"),
    y="network",
    x="mean",
    hue="type",
    order=[
        "Karate Club",
        "Dolphins",
        "Political Books",
        "Citation Network",
        "Political Blogs",
        "TV Shows",
        "Politicians",
    ][::-1],
    # capsize=0.07,
    # hue_order=["Co-evolutionary model", "Network Formation model"],
)
sns.stripplot(
    data=data.query("type != 'Empirical network'"),
    x="mean",
    y="network",
    hue="type",
    dodge=True,
    order=[
        "Karate Club",
        "Dolphins",
        "Political Books",
        "Citation Network",
        "Political Blogs",
        "TV Shows",
        "Politicians",
    ][::-1],
    edgecolor="gray",
    linewidth=1.5,
    jitter=1,
    # color="black",
    alpha=0.8,
    size=5,
    # hue_order=["Co-evolutionary model", "Network Formation model"],
    # palette=sns.cubehelix_palette(8, rot=-0.25, light=0.9),
)
plt.xlabel(r"$O(A,G)$")
plt.ylabel("")
plt.xlim(0.0, 0.5)
# plt.axvline(x=0, color = "black", ls="--")
plt.ylim(-0.5, 6.5)
handles, labels = g.get_legend_handles_labels()

# When creating the legend, only use the first two elements
# to effectively remove the last two.
l = plt.legend(
    handles[0:4], labels[0:4], title="Type of Model", bbox_to_anchor=(1.4, 0.6)
)

g.hlines(
    [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5],
    xmin=0.0,
    xmax=0.5,
    colors="gray",
    linestyles="dotted",
)

plt.savefig("../plots/overall/Model_Evaluation.png", dpi=300, bbox_inches="tight")
plt.savefig("../plots/overall/Model_Evaluation.pdf", dpi=300, bbox_inches="tight")

plt.clf()

pio.renderers.default = "notebook"


def get_important_parameters(network):
    study = joblib.load(f"../analysis/data/optimization/{network}_study.pkl")
    dict_of_values = dict(optuna.importance.get_param_importances(study))
    dict_of_values["network"] = network
    return dict_of_values


def rename_value_names(string):
    old_names = [
        "randomness",
        "tie_dissolution",
        "threshold",
        "negative_learning_rate",
        "positive_learning_rate",
    ]
    new_names = [r"$R$", r"$P(D)$", r"$T$", r"$\beta$", r"$\alpha$"]
    translation = {old_names[i]: new_names[i] for i in range(len(old_names))}
    return translation.get(string)


list_of_parameters = [get_important_parameters(network) for network in NAME_DICTIONARY]

importance_df = pd.DataFrame(list_of_parameters)

importance_df = importance_df.melt(id_vars="network")

importance_df["variable"] = importance_df["variable"].apply(
    lambda x: rename_value_names(x)
)
importance_df["network"] = importance_df["network"].apply(
    lambda x: change_network_labels(x)
)

facet = sns.FacetGrid(importance_df, col="network", height=4)
facet.map(sns.barplot, "variable", "value")
facet.set_xlabels("")
facet.set_ylabels("Parameter Importance")
facet.set_titles(col_template="{col_name}")
facet.savefig(
    "../plots/overall/Parameter_Importance.png",
    dpi=300,
    bbox_inches="tight",
)
facet.savefig(
    "../plots/overall/Parameter_Importance.pdf",
    dpi=300,
    bbox_inches="tight",
)


def get_important_network_characteristics(name: str, network: nx.Graph()):
    network = get_main_component(network=network)
    degrees = np.array([degree[1] for degree in network.degree()])

    return {
        "name": name,
        "nodes": len(degrees),
        "edges": len(network.edges()),
        "mean_degree": degrees.mean(),
        "sd_degree": degrees.std(),
        "clustering coefficient": nx.average_clustering(network),
        "average_path_length": find_average_path(network),
    }


df_network_characteristics = pd.DataFrame(
    [
        get_important_network_characteristics(name, network)
        for name, network in NAME_DICTIONARY.items()
    ]
)


def rename_plot(g, titles):
    for ax, title in zip(g, titles):
        ax.set(xlabel=title)
    return g


study = joblib.load(f"../analysis/data/optimization/best_optimization_results_.pkl")
parameter_values_df = pd.DataFrame(study).round(3)
pio.renderers.default = "notebook_connected"

list_of_names = [
    "Karate Club",
    "Dolphins",
    "Political Books",
    "Citation Network",
    "Political Blogs",
    "TV Shows",
    "Politicians",
]

for name_file, name_list in zip(NAME_DICTIONARY, list_of_names):
    study = joblib.load(f"../analysis/data/optimization/{name_file}_study.pkl")

    fig = optuna.visualization.plot_optimization_history(study)
    fig.update_layout(title={"text": ""})
    # fig.write_image(f"../plots/overall/Optimization_History_{name_file}.png")
    # fig.write_image(
    #     f"../plots/overall/Optimization_History_{name_file}.pdf", format="pdf"
    # )

    fig = optuna.visualization.plot_slice(study)
    fig.update_layout(
        title={"text": ""},
    )
    titles = [r"$\beta$", r"$\alpha$", r"$R$", r"$T$", r"$P(D)$"]
    for i in range(1, 6):
        fig.update_xaxes(col=i, title=titles[i - 1])
    fig.show()
    # fig.write_image(f"../plots/overall/Plot_Slice_{name_file}.png")
    # fig.write_image(f"../plots/overall/Plot_Slice_{name_file}.pdf", format="pdf")
