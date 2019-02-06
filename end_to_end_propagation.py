import numpy as np
import hidden_layer_propagation as hlp
import output_layer_propagation as olp

def initialize_cache(debug_mode=False):
    # if debug_mode:
    #     print("Message: initializing cache...")
    #     print("\tStack trace: end_to_end_propagation.initialize_cache()")
    cache = dict()
    cache["Z"] = dict()
    cache["A"] = dict()
    cache["dA"] = dict()
    cache["dZ"] = dict()
    cache["dW"] = dict()
    cache["db"] = dict()
    return cache

def end_to_end_forward(X, hyperparameters, parameters, cache, debug_mode=False):
    L = hyperparameters["L"]
    cache["A"][0] = X
    for l in range(1, L + 2):
        parameters, cache = hlp.one_layer_forward(hyperparameters=hyperparameters, parameters=parameters, cache=cache, l=l, debug_mode=debug_mode)
    return (parameters, cache)

def end_to_end_backward(Y, hyperparameters, parameters, cache, learning_rate=0.001, debug_mode=False):
    L = hyperparameters["L"]
    cache = olp.compute_dAL(hyperparameters=hyperparameters, Y=Y, cache=cache, debug_mode=debug_mode)
    for l in reversed(range(1, L + 2)):
        parameters, cache = hlp.one_layer_backward(hyperparameters=hyperparameters, parameters=parameters, cache=cache, l=l, learning_rate=learning_rate, debug_mode=debug_mode)
    return (parameters, cache)
