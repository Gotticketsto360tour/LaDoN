from cProfile import label
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(rc={"figure.figsize": (11.7, 8.27)})
sns.set_context("talk")

data = pd.read_csv("data/ESS1-9e01_1/ESS1-9e01_1.csv")

set(data["cntry"])

data = data[(data["essround"] == 9) & (data["lrscale"] < 11)]

grouped_data = (
    data.groupby(["cntry", "lrscale"]).agg(opinion=("dweight", sum)).reset_index()
)

data.groupby("cntry").apply(pd.DataFrame.kurtosis).reset_index()[
    ["cntry", "lrscale"]
].sort_values("lrscale")

for country in set(data["cntry"].values):
    subset = data[data["cntry"] == country]
    sns.displot(
        data=subset,
        x="lrscale",
        discrete=True,
        stat="percent",
        common_norm=False,
        kde=True,
    ).set(xlabel="Left-right scale", title=country)
    plt.savefig(f"plots/{country}.png")

sns.kdeplot(
    data=data, x="lrscale", hue="cntry", common_norm=False, linewidth=0.5, legend=False
)
