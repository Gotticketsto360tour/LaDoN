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

plotting = sns.histplot(
    data=data, x="distances", stat="percent", hue="threshold", kde=True
).set(title="Average distance to neighbor's opinion", xlabel="Average distance")

sns.regplot(data=data, x="degrees", y="agent_number")
sns.regplot(data=data, x="centrality", y="agent_number")
