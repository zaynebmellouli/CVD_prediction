import numpy as np

def split_set(x_train, y_train, ratio=0.8):
    
    # Get the number of samples
    num_samples = x_train.shape[0]
    
    # Generate a random permutation of the indices
    indices = np.random.permutation(num_samples)
    
    # Calculate the split index based on the given ratio
    split_index = int(num_samples * ratio)
    
    # Split the indices into train and test
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    
    # Split the data using the indices
    x_train_split = x_train[train_indices]
    y_train_split = y_train[train_indices]
    
    x_test_split = x_train[test_indices]
    y_test_split = y_train[test_indices]
    
    return x_train_split, x_test_split, y_train_split, y_test_split
