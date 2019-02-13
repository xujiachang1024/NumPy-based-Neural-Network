import numpy as np
import activation_functions as af
import batch_allocation as ba
import end_to_end_propagation as e2ep
import hidden_layer_propagation as hlp
import output_layer_propagation as olp
import matplotlib.pyplot as plt

class NumPyBasedNeuralNetwork(object):

    def __init__(self, L=None, dimensions=None, activations=None, debug_mode=False):
        # declare hyperparameters
        self.__hyperparameters = None
        # declare parameters
        self.__parameters = None
        # declare logs of costs
        self.__costs = None
        # set model architecture
        self.set_architecture(L=L, dimensions=dimensions, activations=activations, debug_mode=debug_mode)

    def set_architecture(self, L=None, dimensions=None, activations=None, debug_mode=False):
        # initialize hyperparameters
        self.__hyperparameters = dict()
        self.__hyperparameters["L"] = L # number of hidden layers
        self.__hyperparameters["dimensions"] = dimensions # number of neurons at each layer
        self.__hyperparameters["activations"] = activations # type of activation function at each layer
        # initialize parameters
        self.__parameters = dict()
        self.__parameters["W"] = dict() # weights
        self.__parameters["b"] = dict() # bias terms
        # reset model parameters
        self.__reset_parameters(debug_mode=debug_mode)

    def __reset_parameters(self, debug_mode=False):
        # randomize parameter values
        if self.__hyperparameters["L"] != None and self.__hyperparameters["dimensions"] != None:
            if (self.__hyperparameters["L"] + 2) == len(self.__hyperparameters["dimensions"]) and (self.__hyperparameters["L"] + 1) == len(self.__hyperparameters["activations"]):
                for l in range(1, self.__hyperparameters["L"] + 2):
                    n_h_curr = self.__hyperparameters["dimensions"][l]
                    n_h_prev = self.__hyperparameters["dimensions"][l - 1]
                    self.__parameters["W"][l] = np.random.randn(n_h_curr, n_h_prev) * np.sqrt(2.0 / n_h_prev)
                    self.__parameters["b"][l] = np.zeros((n_h_curr, 1))
            else:
                if debug_mode:
                    print("Error: inconsistent number of hidden layers")
                    print("\tStack trace: NumPyBasedNeuralNetwork.__reset_parameters()")
        else:
            if debug_mode:
                print("Warning: weights and bias terms not initialized")
                print("\tStack trace: NumPyBasedNeuralNetwork.__reset_parameters()")
        # clear logs of costs
        self.__costs = dict()
        self.__costs["epoch"] = []
        self.__costs["iteration"] = []
        # print dimensions
        if debug_mode:
            self.__print_dimensions(hyperparameters=self.__hyperparameters, parameters=self.__parameters, cache=None)

    def __print_dimensions(self, hyperparameters, parameters=None, cache=None):
        if parameters != None:
            print("Architecture: " + str(hyperparameters["L"]) + " hidden layers")
            print("|" + "{:15s}".format("layer") + "|" + "{:15s}".format("W") + "|" + "{:15s}".format("b") + "|" + "{:15s}".format("activation") + "|")
            for l in range(1, hyperparameters["L"] + 2):
                print("|" + "{:15s}".format(str(l)) + "|" + "{:15s}".format(str(parameters["W"][l].shape)) + "|" + "{:15s}".format(str(parameters["b"][l].shape)) + "|" +
                                                            "{:15s}".format(str(hyperparameters["activations"][l])) + "|")
        if cache != None:
            print("Propagation:")
            print("|" + "{:15s}".format("layer") + "|" + "{:15s}".format("Z") + "|" + "{:15s}".format("A") + "|" +
                                                         "{:15s}".format("dA") + "|" + "{:15s}".format("dZ") + "|" +
                                                         "{:15s}".format("dW") + "|" + "{:15s}".format("db") + "|")
            for l in range(1, hyperparameters["L"] + 2):
                print("|" + "{:15s}".format(str(l)) + "|" + "{:15s}".format(str(cache["Z"][l].shape)) + "|" + "{:15s}".format(str(cache["A"][l].shape)) + "|" +
                                                             "{:15s}".format(str(cache["dA"][l].shape)) + "|" + "{:15s}".format(str(cache["dZ"][l].shape)) + "|" +
                                                             "{:15s}".format(str(cache["dW"][l].shape)) + "|" + "{:15s}".format(str(cache["db"][l].shape)) + "|")


    def fit(self, X, Y, learning_rate=0.001, decay_rate=0.1, early_stopping_point=1000, convergence_tolerance=0.00001, batch_size=1, debug_mode=False, cost_plot_mode=False):
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
        if Y.shape[0] != self.__parameters["W"][self.__hyperparameters["L"] + 1].shape[0]:
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
            self.__parameters, epoch_cache = e2ep.end_to_end_forward(X=X, hyperparameters=self.__hyperparameters, parameters=self.__parameters, cache=epoch_cache, debug_mode=debug_mode)
            # epoch: compute and log cost
            epoch_cost = olp.compute_cost(Y=Y, hyperparameters=self.__hyperparameters, cache=epoch_cache, loss="cross-entropy", debug_mode=debug_mode)
            if debug_mode:
                print("Epoch " + str(epoch) + "\t cost = " + str(epoch_cost))
            self.__costs["epoch"].append(epoch_cost)
            # epoch: check against convergence tolerance
            if epoch >= 2 and abs(self.__costs["epoch"][-1] - self.__costs["epoch"][-2]) < convergence_tolerance:
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
                iterative_cache = e2ep.initialize_cache(debug_mode=debug_mode)
                self.__parameters, iterative_cache = e2ep.end_to_end_forward(X=X_batch, hyperparameters=self.__hyperparameters, parameters=self.__parameters, cache=iterative_cache, debug_mode=debug_mode)
                # batch iteration: compute and log cost
                iterative_cost = olp.compute_cost(Y=Y_batch, hyperparameters=self.__hyperparameters, cache=iterative_cache, loss="cross-entropy", debug_mode=debug_mode)
                self.__costs["iteration"].append(iterative_cost)
                # batch iteration: end-to-end backward propagation
                decayed_learning_rate = (1.0 / (1.0 + decay_rate * epoch)) * learning_rate
                self.__parameters, iterative_cache = e2ep.end_to_end_backward(Y=Y_batch, hyperparameters=self.__hyperparameters, parameters=self.__parameters, cache=iterative_cache, learning_rate=decayed_learning_rate, debug_mode=debug_mode)
                if debug_mode:
                    self.__print_dimensions(hyperparameters=self.__hyperparameters, parameters=None, cache=iterative_cache)
                    # input("Press enter to proceed to next iteration...")
        if cost_plot_mode:
            # plot epoch costs
            plt.plot(self.__costs["epoch"])
            plt.title("NumPy-based Neural Network, batch size = " + str(batch_size) + "\nEpoch cross-entropy costs\nErnest Xu")
            plt.xlabel("Epoch")
            plt.ylabel("Cross-entropy cost")
            plt.show()
            # plot iterative costs
            plt.plot(self.__costs["iteration"])
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
        predicted_cache = e2ep.initialize_cache(debug_mode=debug_mode)
        self.__parameters, predicted_cache = e2ep.end_to_end_forward(X=X, hyperparameters=self.__hyperparameters, parameters=self.__parameters, cache=predicted_cache, debug_mode=debug_mode)
        AL = predicted_cache["A"][self.__hyperparameters["L"] + 1]
        Y_predicted = np.argmax(AL, axis=0)
        return Y_predicted
