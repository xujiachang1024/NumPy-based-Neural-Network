import numpy as np

"""
The sigmoid value in forward propagation

@param Z: the NumPy array of original values, shape = (n_h, m)
@param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is false
@return a NumPy array of sigmoid values in forward propagation
"""
def sigmoid_forward(Z, debug_mode=False):
    if debug_mode:
        print("Message: Z.shape = " + str(Z.shape))
        print("\tStack trace: activation_functions.sigmoid_forward()")
    sigmoid_forward = 1.0 / (1.0 + np.exp(Z))
    return sigmoid_forward

"""
The sigmoid value in backward propagation

@param Z: the NumPy array of the original values, shape = (n_h, m)
@param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is false
@return a NumPy array of sigmoid values in backward propagation
"""
def sigmoid_backward(Z, debug_mode=False):
    if debug_mode:
        print("Message: Z.shape = " + str(Z.shape))
        print("\tStack trace: activation_functions.sigmoid_backward()")
    sigmoid_backward = np.multiply(Z, (1.0 - Z))
    return sigmoid_backward
