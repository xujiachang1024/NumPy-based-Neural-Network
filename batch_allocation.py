import numpy as np

def allocate_batches(X, Y, batch_size, debug_mode=False):
    if X.shape[1] != Y.shape[1]:
        if debug_mode:
            print("Error: inconsistent number of examples")
            print("\tStack trace: batch_allocation.allocate_batches()")
        return None
    # get the number of examples
    m = Y.shape[1]
    if batch_size > m:
        batch_size = m
    elif batch_size < 1:
        batch_size = 1
    num_batches = m // batch_size + (m % batch_size > 0)
    X_batches = np.array_split(X, num_batches, axis=1)
    Y_batches = np.array_split(Y, num_batches, axis=1)
    return (X_batches, Y_batches)
