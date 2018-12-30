import networkx as nx
import numpy as np
import pygsp
import random
import pandas as pd
from tqdm import tqdm

class WaveletMachine:
    """
    The class is a blue print for the procedure described in "Learning Structural Node Embeddings Via Diffusion Wavelets".
    """
    def __init__(self, G, settings):
        """
        This method 
        :param G: Input networkx graph object.
        :param settings: argparse object with settings.
        """
        self.G = pygsp.graphs.Graph(nx.adjacency_matrix(G))
        self.number_of_nodes = len(nx.nodes(G))
        self.settings = settings
        if self.number_of_nodes > self.settings.switch:
            self.settings.mechanism = "approximate"

        self.steps = [x*self.settings.step_size for x in range(self.settings.sample_number)]

    def single_wavelet_generator(self, node):
        """
        Calculating the characteristic function for a given node, using the eigendecomposition.
        :param node: Node that is being embedded.
        """
        impulse = np.zeros((self.number_of_nodes))
        impulse[node] = 1
        wavelet_coefficients = np.dot(np.dot(np.dot(self.eigen_vectors,np.diag(np.exp(-self.settings.heat_coefficient*self.eigen_values))),np.transpose(self.eigen_vectors)), impulse)
        return wavelet_coefficients

    def exact_wavelet_calculator(self):
        """
        Calculates the structural role embedding using the exact eigenvalue decomposition.
        """
        self.real_and_imaginary = []
        for node in tqdm(range(0, self.number_of_nodes)):
            wave = self.single_wavelet_generator(node)
            wavelet_coefficients = [np.mean(np.exp(wave*1*step*1j)) for step in self.steps]
            self.real_and_imaginary.append(wavelet_coefficients)
        self.real_and_imaginary = np.array(self.real_and_imaginary)

    def exact_structural_wavelet_embedding(self):
        """
        Calculates the eigenvectors, eigenvalues and an exact embedding is created.
        """
        self.G.compute_fourier_basis()
        self.eigen_values = self.G.e / max(self.G.e)
        self.eigen_vectors = self.G.U
        self.exact_wavelet_calculator()

    def approximate_wavelet_calculator(self):
        """
        Given the Chebyshev polynomial, graph the approximate embedding is calculated. 
        """
        self.real_and_imaginary = []
        for node in tqdm(range(0,self.number_of_nodes)):
            impulse = np.zeros((self.number_of_nodes))
            impulse[node] = 1
            wavelet_coefficients = pygsp.filters.approximations.cheby_op(self.G, self.chebyshev, impulse)
            self.real_and_imaginary.append([np.mean(np.exp(wavelet_coefficients*1*step*1j)) for step in self.steps])
        self.real_and_imaginary = np.array(self.real_and_imaginary)


    def approximate_structural_wavelet_embedding(self):
        """
        Estimating the largest eigenvalue, setting up the heat filter and the Cheybshev polynomial. Using the approximate wavelet calculator method.
        """
        self.G.estimate_lmax()
        self.heat_filter = pygsp.filters.Heat(self.G, tau=[self.settings.heat_coefficient])
        self.chebyshev = pygsp.filters.approximations.compute_cheby_coeff(self.heat_filter, m = self.settings.approximation)
        self.approximate_wavelet_calculator()

    def create_embedding(self):
        """
        Depending the mechanism setting creating an exact or approximate embedding.
        """
        if self.settings.mechanism == "exact":
            self.exact_structural_wavelet_embedding()
        else:
            self.approximate_structural_wavelet_embedding()

    def transform_and_save_embedding(self):
        """
        Transforming the numpy array with real and imaginary values to a pandas dataframe and saving it as a csv.
        """
        print("\nSaving the embedding.")
        self.real_and_imaginary = np.concatenate([self.real_and_imaginary.real, self.real_and_imaginary.imag], axis = 1)
        columns_1 = ["reals_" + str(x) for x in range(self.settings.sample_number)]
        columns_2 = ["imags_" + str(x) for x in range(self.settings.sample_number)]
        columns = columns_1 + columns_2
        self.real_and_imaginary = pd.DataFrame(self.real_and_imaginary, columns = columns)
        self.real_and_imaginary.to_csv(self.settings.output, index = None)
