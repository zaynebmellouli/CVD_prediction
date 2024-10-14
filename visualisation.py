import numpy as np

def correlation_f(x_train, y_train):
    correlations = []

    # Iterate over each feature in x_train
    for i in range(x_train.shape[1]):
        # Extract the i-th feature from x_train
        x_feature = x_train[..., i]

        # Extract the target column from y_train
        y_target = y_train

        # Create a boolean mask to filter out rows where either x_feature or y_target has NaN
        mask = ~np.isnan(x_feature) & ~np.isnan(y_target)

        # Apply the mask to filter x_feature and y_target
        filtered_x = x_feature[mask]
        filtered_y = y_target[mask]

        # Calculate Pearson correlation only if there are valid (non-NaN) values left
        if len(filtered_x) > 1:  # Need at least two values to calculate correlation
            with np.errstate(invalid='ignore'):
                corr = np.corrcoef(filtered_x, filtered_y)[0, 1]
        else:
            corr = np.nan
        
        # Append the correlation result to the list
        correlations.append(corr)
    
    sorted_correlations = np.argsort(-np.abs(correlations))
    return sorted_correlations, correlations

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
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    