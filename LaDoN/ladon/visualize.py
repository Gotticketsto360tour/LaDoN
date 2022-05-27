from bokeh.io import output_notebook, show, export_png
from bokeh.models import (
    Range1d,
    Circle,
    ColumnDataSource,
    MultiLine,
    EdgesAndLinkedNodes,
    NodesAndLinkedEdges,
)
from bokeh.palettes import Spectral11
from bokeh.plotting import figure
from bokeh.plotting import from_networkx
from bokeh.palettes import (
    Blues8,
    Reds8,
    Purples8,
    Oranges8,
    Viridis8,
    Spectral8,
    Category20_20,
)
from bokeh.transform import linear_cmap
from bokeh.models import EdgesAndLinkedNodes, NodesAndLinkedEdges
from bokeh.plotting import save, output_file, figure
import pandas as pd
import networkx
import matplotlib.pyplot as plt
import numpy as np
import colorcet
from colorcet import b_glasbey_bw_minc_20
from selenium import webdriver

from ladon.classes.network import Network
import random

output_notebook()


def plot_graph(network: Network, plot_type="community", save_path="", title="") -> None:
    """Plot the graph from the Network-class

    Args:
        network (Network): Instance of a Network-class
        plot_type (str, optional): String specifying plot type. Currently only "agent_type" is working as intented. Defaults to "community".
        save_path (str, optional): String specifying the path to save the graph's html object to. When no path is given, the graph is not saved. Defaults to "".
    """
    G = network.graph
    degrees = dict(networkx.degree(G))
    networkx.set_node_attributes(G, name="degree", values=degrees)
    number_to_adjust_by = 5
    adjusted_node_size = dict(
        [(node, degree + number_to_adjust_by) for node, degree in networkx.degree(G)]
    )
    networkx.set_node_attributes(
        G, name="adjusted_node_size", values=adjusted_node_size
    )

    opinions = {i: network.agents.get(i).opinion for i in network.agents}

    networkx.set_node_attributes(G, name="opinions", values=opinions)

    if plot_type == "community":
        communities = networkx.algorithms.community.greedy_modularity_communities(G)
        # Create empty dictionaries
        modularity_class = {}
        modularity_color = {}
        # Loop through each community in the network
        for community_number, community in enumerate(communities):
            # For each member of the community, add their community number and a distinct color
            for name in community:
                modularity_class[name] = community_number
                modularity_color[name] = b_glasbey_bw_minc_20[community_number]

        networkx.set_node_attributes(
            G, name="modularity_class", values=modularity_class
        )
        networkx.set_node_attributes(
            G, name="modularity_color", values=modularity_color
        )
        color_by_this_attribute = "modularity_color"
        HOVER_TOOLTIPS = [
            ("Character", "@index"),
            ("Degree", "@degree"),
            ("Modularity Class", "@modularity_class"),
            ("Modularity Color", "$color[swatch]:modularity_color"),
            ("Opinion", "$opinions{%0.2f}"),
        ]
    elif plot_type == "agent_type":
        color_by_this_attribute = "opinions"
        HOVER_TOOLTIPS = [
            ("Degree", "@degree"),
            ("Opinion", "@opinions{0.00}"),
        ]
    # Choose colors for node and edge highlighting
    node_highlight_color = "white"
    edge_highlight_color = "black"

    # Choose attributes from G network to size and color by — setting manual size (e.g. 10) or color (e.g. 'skyblue') also allowed
    size_by_this_attribute = "adjusted_node_size"

    # Pick a color palette — Blues8, Reds8, Purples8, Oranges8, Viridis8
    color_palette = Blues8

    # Choose a title!
    if not title:
        title = "LaDoN Network"

    # Establish which categories will appear when hovering over each node

    # Create a plot — set dimensions, toolbar, and title
    plot = figure(
        tooltips=HOVER_TOOLTIPS,
        tools="pan,wheel_zoom,save,reset",
        active_scroll="wheel_zoom",
        x_range=Range1d(-10.1, 10.1),
        y_range=Range1d(-10.1, 10.1),
        title=title,
    )

    # Create a network graph object
    # https://networkx.github.io/documentation/networkx-1.9/reference/generated/networkx.drawing.layout.spring_layout.html
    network_graph = from_networkx(G, networkx.spring_layout, scale=10, center=(0, 0))

    # Set node sizes and colors according to node degree (color as category from attribute)
    network_graph.node_renderer.glyph = Circle(
        size=size_by_this_attribute,
        fill_color=linear_cmap(
            color_by_this_attribute,
            "Blues256",
            -1,
            1,
        ),
    )
    # Set node highlight colors
    network_graph.node_renderer.hover_glyph = Circle(
        size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2
    )
    network_graph.node_renderer.selection_glyph = Circle(
        size=size_by_this_attribute, fill_color=node_highlight_color, line_width=2
    )

    # Set edge opacity and width
    network_graph.edge_renderer.glyph = MultiLine(line_alpha=0.5, line_width=1)
    # Set edge highlight colors
    network_graph.edge_renderer.selection_glyph = MultiLine(
        line_color=edge_highlight_color, line_width=2
    )
    network_graph.edge_renderer.hover_glyph = MultiLine(
        line_color=edge_highlight_color, line_width=2
    )

    # Highlight nodes and edges
    network_graph.selection_policy = NodesAndLinkedEdges()
    network_graph.inspection_policy = NodesAndLinkedEdges()

    plot.grid.visible = False
    plot.axis.visible = False

    plot.renderers.append(network_graph)

    show(plot)

    if save_path:
        driver = webdriver.Firefox()
        export_png(
            obj=plot,
            filename=save_path,
            webdriver=driver,  # width=2000, height=2000
        )
        driver.close()
        # save(plot)


def generate_network_plots(
    network: Network, plot_type="agent_type", save_path="", title="", run=0
) -> None:
    # random.seed(run)
    # np.random.seed(run)
    if save_path:
        plot_graph(
            network=network,
            plot_type=plot_type,
            save_path=f"{save_path}_0.png",
            title="Timestep: 0",
        )
    else:
        plot_graph(
            network=network,
            plot_type=plot_type,
            save_path="",
            title="Timestep: 0",
        )
    for run in range(1, 21):
        for turn in range(1, 501):
            network.take_turn()
        if save_path:
            plot_graph(
                network=network,
                plot_type=plot_type,
                title=f"Timestep: {turn * run}",
                save_path=f"{save_path}_{run * turn}.png",
            )
        else:
            plot_graph(
                network=network,
                plot_type=plot_type,
                title=f"Timestep: {turn * run}",
                save_path="",
            )
