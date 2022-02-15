from cProfile import label
import pandas as pd
import seaborn as sns

data = pd.read_csv("data/ESS1-9e01_1/ESS1-9e01_1.csv")

set(data["cntry"])

data = data[(data["essround"] == 9) & (data["lrscale"] < 11)]

grouped_data = (
    data.groupby(["cntry", "lrscale"]).agg(opinion=("dweight", sum)).reset_index()
)

data.groupby("cntry").apply(pd.DataFrame.kurtosis).reset_index()[
    ["cntry", "lrscale"]
].sort_values("lrscale")

sns.displot(
    data=data,
    x="lrscale",
    row="cntry",
    discrete=True,
    stat="percent",
    common_norm=False,
    kde=True,
)

sns.kdeplot(data=data, x="lrscale", hue="cntry", common_norm=False, linewidth=0.5)
