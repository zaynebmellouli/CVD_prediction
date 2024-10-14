import numpy as np

def remove_uncorrelated_feature(x_train, x_test, correlations):
    most_correlated_features = []
    
    select = []
    
    for idx, i in enumerate(correlations):
        #print(i)
        if np.abs(i) > 0.1:
            select.append(idx)
            
    return x_train[..., select], x_test[..., select]

def remove_unsuffisant_data_feature(x_train, x_test, threshold=0.05):
    nan_percentage = np.mean(np.isnan(x_train), axis=0)
    valid_features = nan_percentage <= threshold
    
    return x_train[:,valid_features], x_test[:,valid_features]
            
            
            
            
            
