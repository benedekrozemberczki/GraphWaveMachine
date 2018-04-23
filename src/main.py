import pandas as pd
import networkx as nx
from spectral_machinery import WaveletMachine
from parser import parameter_parser

def read_graph(path):
    """
    Reading the edge list from the path and returning the networkx graph object.
    :param path: Path to the edge list.
    :return graph: Graph from edge list.
    """
    edge_list = pd.read_csv(path).values.tolist()
    graph = nx.from_edgelist(edge_list)
    return graph

if __name__ == "__main__":
    settings = parameter_parser()
    G = read_graph(settings.input)
    machine = WaveletMachine(G,settings)
    machine.create_embedding()
    machine.transform_and_save_embedding()
