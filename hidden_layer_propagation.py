import numpy as np
import activation_functions as af

def one_layer_forward(parameters, cache, l, activation="sigmoid", debug_mode=False):
    W = parameters["W"][l]
    b = parameters["b"][l]
    A_prev = cache["A"][l - 1]
    # check the number of neurons at the (l-1)-th layer
    if W.shape[1] != A_prev.shape[0]:
        if debug_mode:
            print("Error: inconsistent dimensions of W" + str(l) + " & A" + str(l - 1))
            print("\tStack trace: hidden_layer_forward.nonlinear_forward()")
        return None
    # check the number of neurons at the l-th layer
    if W.shape[0] != b.shape[0]:
        if debug_mode:
            print("Error: inconsistent dimensions of W" + str(l) + " & b" + str(l))
            print("\tStack trace: hidden_layer_forward.nonlinear_forward()")
        return None
    # calculate linear forward
    cache["Z"][l] = np.dot(W, A_prev) + b
    # activate nonlinear forward: sigmoid
    if activation.lower() == "sigmoid":
        cache["A"][l] = af.sigmoid_forward(Z, debug_mode=debug_mode)
        return (parameters, cache)
    return None

def compute_gradients(parameters, cache, l, activation="sigmoid", debug_mode=False):
    W = parameters["W"[l]
    Z = cache["Z"][l]
    A_prev = cache["A"][l - 1]
    dA = cache["dA"][l]
    # get the gradient of Z
    if activation.lower()="sigmoid":
        cache["dZ"][l] = np.multiply(dA, af.sigmoid_backward(Z))
    else:
        if debug_mode:
            print("Error: unsupported activation function")
            print("\tStack trace: hidden_layer_propagation.nonlinear_backward()")
        return None
    # get the number of examples
    m = A_prev.shape[1]
    # get the gradient of W
    cache["dW"][l] = (1.0 / m) * np.dot(dZ, A_prev.T)
    # get the gradient of b
    cache["db"][l] = (1.0 / m) * np.sum(dZ, axis=1, keepdims=True)
    # get the gradient of A_prev
    cache["dA"][l - 1] = np.dot(W.T, dZ)
    return (parameters, cache)

def update_parameters(parameters, cache, l, learning_rate=0.001, debug_mode=False):
    dW = cache["dW"][l]
    db = cache["db"][l]
    if parameters["W"][l].shape != dW.shape:
        if debug_mode:
            print("Error: inconsistent dimensions of W" + str(l) + " & dW"+ str(l))
            print("\tStack trace: hidden_layer_propagation.update_parameters()")
        return None
    if parameters["b"][l].shape != db.shape:
        if debug_mode:
            print("Error: inconsistent dimensions of b" + str(l) + " & db"+ str(l))
            print("\tStack trace: hidden_layer_propagation.update_parameters()")
        return None
    parameters["W"][l] -= learning_rate * dW
    parameters["b"][l] -= learning_rate * db
    return (parameters, cache)

def one_layer_backward(parameters, cache, l, activation="sigmoid", learning_rate=0.001, debug_mode=False):
    parameters, cache = compute_gradients(parameters=parameters, cache=cache, l=l, activation=activation, debug_mode=debug_mode)
    parameters, cache = update_parameters(parameters=parameters, cache=cache, l=l, learning_rate=learning_rate, debug_mode=debug_mode)
    return (parameters, cache)
