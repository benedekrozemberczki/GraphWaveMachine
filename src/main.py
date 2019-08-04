import pandas as pd
import networkx as nx
from texttable import Texttable
from parser import parameter_parser
from spectral_machinery import WaveletMachine

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    tab = Texttable() 
    tab.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(tab.draw())

def read_graph(settings):
    """
    Reading the edge list from the path and returning the networkx graph object.
    :param path: Path to the edge list.
    :return graph: Graph from edge list.
    """
    if settings.edgelist_input:
        graph = nx.read_edgelist(settings.input)
    else:
        edge_list = pd.read_csv(settings.input).values.tolist()
        graph = nx.from_edgelist(edge_list)
        graph.remove_edges_from(graph.selfloop_edges())
    return graph

if __name__ == "__main__":
    settings = parameter_parser()
    tab_printer(settings)
    G = read_graph(settings)
    machine = WaveletMachine(G,settings)
    machine.create_embedding()
    machine.transform_and_save_embedding()
