from numpy.core.numeric import outer
import networkx as nx

THRESHOLDS = [0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2]
RANDOMNESS = [0.1]
POSITIVE_LEARNING_RATES = [0.05, 0.1, 0.15, 0.2, 0.25]
NEGATIVE_LEARNING_RATES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
TIE_DISSOLUTIONS = [0.0, 0.2, 0.4, 0.6, 0.8, 1]

AGENT_CONFIG = {
    "opinion": {"mu": 0, "sigma": 1, "vector_size": 1},
    "type": 1,
}
AGENT_CONFIG_2 = {
    "opinion": {"mu": 2, "sigma": 1, "vector_size": 1},
    "type": 2,
}
AGENT_CONFIG_3 = {
    "opinion": {"mu": 0, "sigma": 0.5, "vector_size": 1},
    "type": 3,
}

CONFIGURATIONS = [AGENT_CONFIG, AGENT_CONFIG_3]  # AGENT_CONFIG_3

CONFIGS = {i: CONFIG for i, CONFIG in enumerate(CONFIGURATIONS)}


facebook = nx.read_edgelist(
    "analysis/data/facebook_combined.txt", create_using=nx.Graph(), nodetype=int
)

astrophysics = nx.read_edgelist(
    "analysis/data/dimacs10-astro-ph/out.dimacs10-astro-ph",
    create_using=nx.Graph(),
    nodetype=int,
)

theoretical_physics = nx.read_edgelist(
    "analysis/data/physics/ca-HepTh.txt",
    create_using=nx.Graph(),
    nodetype=int,
)

with open("analysis/data/fb-pages-government/fb-pages-government.nodes", "rb+") as f:
    data = [str(node, "utf-8").strip().split(",")[-1] for node in f.readlines()[1:]]

government = nx.Graph()

government.add_nodes_from(data)

with open("analysis/data/fb-pages-government/fb-pages-government.edges", "rb+") as f:
    data = [str(node, "utf-8").strip().split(",") for node in f.readlines()]

government.add_edges_from(data)

with open("analysis/data/fb-pages-politician/fb-pages-politician.nodes", "rb+") as f:
    data = [str(node, "utf-8").strip().split(",")[-1] for node in f.readlines()[1:]]

politicians = nx.Graph()

politicians.add_nodes_from(data)

with open("analysis/data/fb-pages-politician/fb-pages-politician.edges", "rb+") as f:
    data = [str(node, "utf-8").strip().split(",") for node in f.readlines()]

politicians.add_edges_from(data)

netscience = nx.read_gml(path="analysis/data/netscience/netscience.gml")

karate = nx.karate_club_graph()

polblogs = nx.read_gml(
    path="analysis/data/polblogs/polblogs.gml",
)
polblogs = nx.Graph(polblogs.to_undirected())

polbooks = nx.read_gml(
    path="analysis/data/polbooks/polbooks.gml",
)

dolphin = nx.read_gml(path="analysis/data/dolphins/dolphins.gml")

NAME_DICTIONARY = {
    "karate": karate,
    "dolphin": dolphin,
    "polbooks": polbooks,
    "netscience": netscience,
    "polblogs": polblogs,
    # "facebook": facebook,
    "politicians": politicians,
    "government": government,
}
