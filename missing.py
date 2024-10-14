import numpy as np

def fill_missing_value(x_train, x_test):
     # Calculate the mean of each feature (dimension) from the training data, ignoring NaNs
    means = np.nanmean(x_train, axis=0)
    
    # Find the indices where there are NaNs in x_train and x_test
    nan_indices_train = np.isnan(x_train)
    nan_indices_test = np.isnan(x_test)
    
    # Replace NaNs in x_train with the corresponding feature mean
    x_train[nan_indices_train] = np.take(means, np.where(nan_indices_train)[1])
    
    # Replace NaNs in x_test with the corresponding feature mean from the training data
    x_test[nan_indices_test] = np.take(means, np.where(nan_indices_test)[1])
    
    return x_train, x_test
