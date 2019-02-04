import numpy as np
import activation_functions as af

def one_layer_forward(hyperparameters, parameters, cache, l, debug_mode=False):
    activation = hyperparameters["activations"][l]
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
        cache["A"][l] = af.sigmoid_forward(Z=Z, debug_mode=debug_mode)
    # activate nonlinear forward: tanh
    elif activation.lower() == "tanh":
        cache["A"][l] = af.tanh_forward(Z=Z, debug_mode=debug_mode)
    # activate nonlinear forward: relu
    elif activation.lower() == "relu":
        cache["A"][l] = af.relu_forward(Z=Z, debug_mode=debug_mode)
    # activate nonlinear forward: leaky relu
    elif activation.lower() == "leaky relu":
        cache["A"][l] = af.leaky_relu_forward(Z=Z, debug_mode=debug_mode)
    else:
        if debug_mode:
            print("Error: unsupported activation function")
            print("\tStack trace: hidden_layer_propagation.nonlinear_forward()")
        return None
    return (parameters, cache)

def compute_gradients(hyperparameters, parameters, cache, l, debug_mode=False):
    activation = hyperparameters["activations"][l]
    W = parameters["W"][l]
    Z = cache["Z"][l]
    A = cache["A"][l]
    A_prev = cache["A"][l - 1]
    dA = cache["dA"][l]
    # get the gradient of Z
    if activation.lower() == "sigmoid":
        cache["dZ"][l] = np.multiply(dA, af.sigmoid_backward(Z=Z, A=A, debug_mode=debug_mode))
    elif activation.lower() == "tanh":
        cache["dZ"][l] = np.multiply(dA, af.tanh_backward(Z=Z, A=A, debug_mode=debug_mode))
    elif activation.lower() == "relu":
        cache["dZ"][l] = np.multiply(dA, af.relu_backward(Z=Z, A=A, debug_mode=debug_mode))
    elif activation.lower() == "leaky tanh":
        cache["dZ"][l] = np.multiply(dA, af.leaky_relu_backward(Z=Z, A=A, debug_mode=debug_mode))
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

def update_parameters(hyperparameters, parameters, cache, l, learning_rate=0.001, debug_mode=False):
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

def one_layer_backward(hyperparameters, parameters, cache, l, learning_rate=0.001, debug_mode=False):
    parameters, cache = compute_gradients(hyperparameters=hyperparameters, parameters=parameters, cache=cache, l=l, debug_mode=debug_mode)
    parameters, cache = update_parameters(hyperparameters=hyperparameters, parameters=parameters, cache=cache, l=l, learning_rate=learning_rate, debug_mode=debug_mode)
    return (parameters, cache)
