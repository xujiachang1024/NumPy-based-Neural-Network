import numpy as np
import activate_functions as af
import batch_allocation as ba
import end_to_end_propagation as e2ep
import hidden_layer_propagation as hlp
import output_layer_propagation as olp
import matplotlib.pyplot as plt

class NumPyBasedNeuralNetwork(object):

    def __init__(self, L=None, dimensions=None, debug_mode=False):
        # declare member variables
        self.__L = None
        self.__dimensions = None
        self.__parameters = None
        self.__epoch_costs = None
        self.__iterative_costs = None
        # set model architecture
        self.set_architecture(L=L, dimensions=dimensions, debug_mode=debug_mode)

    def set_architecture(self, L=None, dimensions=None, debug_mode=False):
        # initialize member variables
        self.__L = L
        self.__dimensions = dimensions
        self.__parameters = dict()
        self.__parameters["W"] = dict()
        self.__parameters["b"] = dict()
        # reset model parameters
        self.__reset_parameters(debug_mode=debug_mode)

    def __reset_parameters(self, debug_mode=False):
        # randomize parameter values
        if self.__L != None and self.__dimensions != None:
            if self.__L + 2 == len(self.__dimensions):
                for l in range(1, self.__L + 2):
                    n_h_curr = self.__dimensions[l]
                    n_h_prev = self.__dimensions[l - 1]
                    self.__parameters["W"][l] = np.random.randn((n_h_curr, n_h_prev))
                    self.__parameters["b"][l] = np.random.randn((n_h_curr, 1))
            else:
                if debug_mode:
                    print("Error: inconsistent number of hidden layers")
                    print("\tStack trace: NumPyBasedNeuralNetwork.__reset_parameters()")
        else:
            if debug_mode:
                print("Warning: weights and bias terms not initialized")
                print("\tStack trace: NumPyBasedNeuralNetwork.__reset_parameters()")
        # clear the log of costs
        self.__epoch_costs = []
        self.__iterative_costs = []

    def fit(self, X, Y, activation="sigmoid", learning_rate=0.001, decay_rate=0.1, early_stopping_point=1000, convergence_tolerance=0.001, batch_size=1, debug_mode=False, cost_plot_mode=True):
        # reset model parameters
        self.__reset_parameters(debug_mode=debug_mode)
        # check the number of examples
        if X.shape[1] != Y.shape[1]:
            if debug_mode:
                print("Error: inconsistent number of examples")
                print("\tStack trace: NumPyBasedNeuralNetwork.fit()")
            return False
        # check the number of features
        if X.shape[0] != self.__parameters["W"][1].shape[1]:
            if debug_mode:
                print("Error: inconsistent number of features")
                print("\tStack trace: NumPyBasedNeuralNetwork.fit()")
            return False
        # check the number of labels
        if Y.shape[0] != self.__parameters["W"][self.__L + 1].shape[0]:
            if debug_mode:
                print("Error: inconsistent number of labels")
                print("\tStack trace: NumPyBasedNeuralNetwork.fit()")
            return False
        # allocate batches
        num_batches, X_batches, Y_batches = ba.allocate_batches(X=X, Y=Y, batch_size=batch_size, debug_mode=debug_mode)
        # epoches of gradient descent
        for epoch in range(early_stopping_point):
            # epoch: end-to-end forward propagation
            epoch_cache = e2ep.initialize_cache(debug_mode=debug_mode)
            self.__parameters, epoch_cache = e2ep.end_to_end_forward(X=X, parameters=self.__parameters, cache=epoch_cache, L=self.__L, activation=activation, debug_mode=debug_mode)
            # epoch: compute and log cost
            epoch_cost = olp.compute_cost(Y=Y, cache=epoch_cache, L=self.__L, loss="cross-entropy", debug_mode=debug_mode)
            if debug_mode:
                print("Epoch " + str(epoch) + "\t cost = " + str(epoch_cost))
            self.__epoch_costs.append(epoch_cost)
            # epoch: check against convergence tolerance
            if epoch >= 2 and abs(self.__epoch_costs[-1] - self.__epoch_costs[-2]) < convergence_tolerance:
                if debug_mode:
                    print("Message: convergence tolerance reached at epoch " + str(epoch))
                    print("\tStack trace: NumPyBased0hlNeuralNetwork.fit()")
                break
            # iterate through batches
            for batch_index in range(num_batches):
                # batch iteration: get the batch based on batch index
                X_batch = X_batches[batch_index]
                Y_batch = Y_batches[batch_index]
                # batch iteration: end-to-end forward propagation
                iterative_cache = e2ep.initialize(debug_mode=debug_mode)
                self.__parameters, iterative_cache = e2ep.end_to_end_forward(X=X_batch, parameters=self.__parameters, cache=iterative_cache, L=self.__L, activation=activation, debug_mode=debug_mode)
                # batch iteration: compute and log cost
                iterative_cost = olp.compute_cost(Y=Y_batch, cache=iterative_cache, L=self.__L, loss="cross-entropy", debug_mode=debug_mode)
                self.__iterative_costs.append(iterative_cost)
                # batch iteration: end-to-end backward propagation
                self.__parameters, iterative_cache = e2ep.end_to_end_backward(Y=Y_batch, parameters=self.__parameters, cache=iterative_cache, L=self.__L, activation=activation, learning_rate=learning_rate, debug_mode=debug_mode)
        if cost_plot_mode:
            # plot epoch costs
            plt.plot(self.__epoch_costs)
            plt.title("NumPy-based Neural Network, batch size = " + str(batch_size) + "\nEpoch cross-entropy costs\nErnest Xu")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-entropy cost")
            plt.show()
            # plot iterative costs
            plt.plot(self.__iterative_costs)
            plt.title("NumPy-based Neural Network, batch size = " + str(batch_size) + "\nIterative cross-entropy costs\nErnest Xu")
            plt.xlabel("Iteration")
            plt.ylabel("Cross-entropy cost")
            plt.show()
        return True

    def predict(self, X, debug_mode=False):
        # check the number of features
        if X.shape[0] != self.__parameters["W"][1].shape[1]:
            if debug_mode:
                print("Error: inconsistent number of features")
                print("\tStack trace: NumPyBasedNeuralNetwork.fit()")
            return None
        predicted_cache = e2ep.initialize(debug_mode=debug_mode)
        self.__parameters, predicted_cache = e2ep.end_to_end_forward(X=X, parameters=self.__parameters, cache=predicted_cache, activation="sigmoid", debug_mode=debug_mode)
        AL = predicted_cache["A"][self.__L + 1]
        predicted_classes = np.argmax(AL, axis=0)
        predicted_onehots = np.zeros(AL.shape)
        for col in range(A.shape[1]):
            predicted_onehots[predicted_classes[col], col] = 1
        return (predicted_classes, predicted_onehots)
