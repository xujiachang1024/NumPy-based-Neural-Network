import random
import numpy as np

def allocate_batches(X, Y, batch_size, debug_mode=False):
    if X.shape[1] != Y.shape[1]:
        if debug_mode:
            print("Error: inconsistent number of examples")
            print("\tStack trace: batch_allocation.allocate_batches()")
        return None
    # get the number of examples
    m = Y.shape[1]
    # validate batch size
    if batch_size > m:
        batch_size = m
    elif batch_size < 1:
        batch_size = 1
    # calculate the number of mini-batches
    num_batches = m // batch_size + (m % batch_size > 0)
    # put indices of examples randomly without replacement into different mini-batches
    index_pool = [i for i in range(m)]
    index_batches = []
    for i in range(num_batches - 1):
        index_batch = []
        for j in range(m // num_batches):
            index_chosen = random.choice(index_pool)
            index_batch.append(index_chosen)
            index_pool.remove(index_chosen)
        index_batches.append(index_batch)
    index_batches.append(index_pool)
    # put examples into different mini-batches based on index_batches
    X_batches = []
    Y_batches = []
    for index_batch in index_batches:
        X_batches.append(X[:, index_batch])
        Y_batches.append(Y[:, index_batch])
    return (num_batches, X_batches, Y_batches)
