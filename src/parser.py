import argparse

def parameter_parser():

    """
    A method to parse up command line parameters.
    """
    parser = argparse.ArgumentParser(description = "Run GraphWave.")

    parser.add_argument("--mechanism",
                        nargs = "?",
                        default = "exact",
	                help = "Eigenvalue calculation method. Default is exact.")

    parser.add_argument("--input",
                        nargs = "?",
                        default = "./data/food_edges.csv",
	                help = "Path to the graph edges. Default is food_edges.csv.")

    parser.add_argument("--output",
                        nargs = "?",
                        default = "./output/embedding.csv",
	                help = "Path to the structural embedding. Default is embedding.csv.")

    parser.add_argument("--heat-coefficient",
                        type = float,
                        default = 1000.0,
	                help = "Heat kernel exponent. Default is 1000.0.")

    parser.add_argument("--sample-number",
                        type = int,
                        default = 50,
	                help = "Number of characteristic function sample points. Default is 50.")

    parser.add_argument("--approximation",
                        type = int,
                        default = 100,
	                help = "Number of Chebyshev approximation. Default is 100.")

    parser.add_argument("--step-size",
                        type = int,
                        default = 20,
	                help = "Number of steps. Default is 20.")

    parser.add_argument("--switch",
                        type = int,
                        default = 100,
	                help = "Number of dimensions. Default is 100.")

    parser.add_argument("--node-label-type",
                        type = str,
                        default = "int",
                        help = "Used for sorting index of output embedding. One of 'int', 'string', or 'float'. Default is 'int'")

    parser.add_argument("--edgelist-input",
                        action = 'store_true',
                        help = "Use NetworkX's edgelist format for input instead of CSV. Default is False")
	
    return parser.parse_args()
