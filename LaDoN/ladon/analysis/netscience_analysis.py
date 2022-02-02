import networkx as nx
import seaborn as sns

sns.set(rc={"figure.figsize": (11.7, 8.27)})

g = nx.read_gml(path="data/netscience/netscience.gml")

nx.algorithms.cluster.average_clustering(g)
nx.algorithms.assortativity.degree_assortativity_coefficient(g)
degrees = [x[1] for x in list(g.degree())]

sns.histplot(degrees, stat="percent")
