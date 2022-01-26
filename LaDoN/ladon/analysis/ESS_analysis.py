import pandas as pd
import seaborn as sns

data = pd.read_csv("data/ESS1-9e01_1/ESS1-9e01_1.csv")

set(data["cntry"])

data = data[(data["essround"] == 9) & (data["lrscale"] < 11)]

grouped_data = (
    data.groupby(["cntry", "lrscale"]).agg(opinion=("dweight", sum)).reset_index()
)

denmark = grouped_data.query("cntry == 'DK'")

sns.barplot(data=denmark, x="lrscale", y="opinion")

sns.set(rc={"figure.figsize": (11.7, 8.27)})
sns.barplot(x=denmark["lrscale"], y=denmark["opinion"])
