import numpy as np

def compute_cost(Y, hyperparameters, cache, loss="cross-entropy", debug_mode=False):
    L = hyperparameters["L"]
    AL = cache["A"][L + 1]
    # check the dimensions of Y & AL
    if Y.shape != AL.shape:
        if debug_mode:
            print("Error: inconsistent dimensions of actual and predicted outputs")
            print("\tY.shape: \t" + str(Y.shape))
            print("\tAL.shape: \t" + str(AL.shape))
            print("\tStack trace: output_layer_propagation.get_cost()")
        return None
    # get the number of examples
    m = Y.shape[1]
    if loss.lower() == "cross-entropy":
        loss = - np.multiply(Y, np.log(AL)) - np.multiply((1.0 - Y), np.log(1.0 - AL))
        cost = np.squeeze((1.0 / m) * np.sum(loss))
        return cost
    if debug_mode:
        print("Error: unsupported type of loss calculation")
        print("\tStack trace: output_layer_propagation.get_cost()")
    return None

def compute_dAL(Y, hyperparameters, cache, debug_mode=False):
    L = hyperparameters["L"]
    AL = cache["A"][L + 1]
    # check the dimensions of Y & AL
    if Y.shape != AL.shape:
        if debug_mode:
            print("Error: inconsistent dimensions of actual and predicted outputs")
            print("\tStack trace: output_layer_propagation.get_dAL()")
        return None
    # get the gradient of AL
    cache["dA"][L + 1] = - np.divide(Y, AL) - np.divide((1 - Y), (1 - AL))
    return cache
