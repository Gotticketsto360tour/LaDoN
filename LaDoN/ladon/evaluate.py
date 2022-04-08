import pickle as pkl
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

sns.set(rc={"figure.figsize": (11.7, 8.27)})
sns.set_context("talk")
blue_pallette = sns.dark_palette("#69d", reverse=True, as_cmap=True)


def change_labels(string: str):
    if string == "Opinion_Model":
        return "With Opinion Dynamics"
    if string == "No_Opinion_Model":
        return "Without Opinion Dynamics"
    else:
        return string


def change_network_labels(string: str):
    translation = {
        "dolphin": "Dolphins",
        "karate": "Karate Club",
        "netscience": "Citation Network",
        "polblogs": "Political Blogs",
        "polbooks": "Political Books",
        "politicians": "Politicians",
    }
    return translation.get(string)


with open(
    f"analysis/data/optimization/data_from_all_runs.pkl",
    "rb",
) as handle:
    data = pkl.load(handle)

data = pd.concat([pd.DataFrame(x) for x in data])

data["type"] = data["type"].apply(lambda x: change_labels(x))
data["network"] = data["network"].apply(lambda x: change_network_labels(x))
g = sns.barplot(
    data=data,
    y="network",
    x="clustering",
    hue="type",
    order=[
        "Dolphins",
        "Karate Club",
        "Citation Network",
        "Political Books",
        "Politicians",
        "Political Blogs",
    ],
)
g.set(xlabel="Average Clustering Coefficient", ylabel="")
plt.legend(
    title="Network",
    bbox_to_anchor=(
        1.45,
        0.6,
    ),
)

g = sns.barplot(
    data=data,
    y="network",
    x="average_path",
    hue="type",
    order=[
        "Dolphins",
        "Karate Club",
        "Citation Network",
        "Political Books",
        "Politicians",
        "Political Blogs",
    ],
)
g.set(xlabel="Average Path Length", ylabel="")
plt.legend(
    title="Network",
    bbox_to_anchor=(
        1.45,
        0.6,
    ),
)
g = sns.barplot(
    data=data.query("type != 'Target'"),
    y="network",
    x="JSD",
    hue="type",
    order=[
        "Dolphins",
        "Karate Club",
        "Citation Network",
        "Political Books",
        "Politicians",
        "Political Blogs",
    ],
)
g.set(xlabel="JSD", ylabel="")
plt.legend(
    title="Network",
    bbox_to_anchor=(
        1.45,
        0.6,
    ),
)

g = sns.barplot(
    data=data.query("type != 'Target'"),
    y="network",
    x="mean",
    hue="type",
    order=[
        "Dolphins",
        "Karate Club",
        "Citation Network",
        "Political Books",
        "Politicians",
        "Political Blogs",
    ],
)
g.set(xlabel="Mean Difference", ylabel="")
plt.legend(title="Network", bbox_to_anchor=(0.5, 0.5, 0.52, 0.52))
plt.savefig("plots/overall/Model_Evaluation.png", dpi=300, bbox_inches="tight")
