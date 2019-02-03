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
@param A: the NumPy array of the activated values, shape = (n_h, m)
@param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is false
@return a NumPy array of sigmoid values in backward propagation
"""
def sigmoid_backward(Z, A, debug_mode=False):
    if debug_mode:
        print("Message: Z.shape = " + str(Z.shape))
        print("\tStack trace: activation_functions.sigmoid_backward()")
    sigmoid_backward = np.multiply(A, (1.0 - A))
    return sigmoid_backward

"""
The tanh value in forward propagation

@param Z: the NumPy array of original values, shape = (n_h, m)
@param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is false
@return a NumPy array of tanh values in forward propagation
"""
def tanh_forward(Z, debug_mode=False):
    if debug_mode:
        print("Message: Z.shape = " + str(Z.shape))
        print("\tStack trace: activation_functions.tanh_forward()")
    tanh_forward = (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    return tanh_forward

"""
The tanh value in backward propagation

@param Z: the NumPy array of the original values, shape = (n_h, m)
@param A: the NumPy array of the activated values, shape = (n_h, m)
@param debug_mode: (optional) a boolean value that indicates whether the debug mode is active; the default value is false
@return a NumPy array of tanh values in backward propagation
"""
def tanh_backward(Z, A, debug_mode=False):
    if debug_mode:
        print("Message: Z.shape = " + str(Z.shape))
        print("\tStack trace: activation_functions.tanh_backward()")
    tanh_backward = 1 - np.square(A)
    return tanh_backward
