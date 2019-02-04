import numpy as np
from numpy_based_neural_network import NumPyBasedNeuralNetwork

def main(debug_mode=True, cost_plot_mode=True):

    # set model architecture
    L = 5
    dimensions = {0: 25, 1: 10, 2: 9, 3: 8, 4:7, 5:6, 6:4}
    activations = {1: "relu", 2: "relu", 3: "relu", 4: "tanh", 5: "tanh", 6: "sigmoid"}
    neural_network = NumPyBasedNeuralNetwork(L=L, dimensions=dimensions, activations=activations, debug_mode=debug_mode)

main()
