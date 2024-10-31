# bare minimum to generate the final_submission.csv file (no cross-validation, no visualiation, no hyperparameter tuning etc)

import numpy as np
from implementations import *

print("loading data...")

x_train = np.genfromtxt("data/x_train.csv", delimiter=",", skip_header=1)
features = np.genfromtxt("data/x_train.csv", delimiter=",", dtype=str, max_rows=1)
y_train = np.genfromtxt("data/y_train.csv", delimiter=",", skip_header=1)
y_features = np.genfromtxt("data/y_train.csv", delimiter=",", dtype=str, max_rows=1)

x_test = np.genfromtxt("data/x_test.csv", delimiter=",", skip_header=1)

print("data loaded")


def to_categorical(array, range_min, range_max, n_bins):
    """
    Converts a numerical array into categorical bins based on specified range and number of bins.
    Parameters:
    array (numpy.ndarray): The input array containing numerical values.
    range_min (float): The minimum value of the range to consider for binning.
    range_max (float): The maximum value of the range to consider for binning.
    n_bins (int): The number of bins to divide the range into.
    Returns:
    function: A function that takes a value and returns the corresponding bin index or the value itself if it is outside the specified range.
              If the value is NaN, it returns -1.
    Notes:
    - Values outside the specified range are returned as is.
    - NaN values are assigned a bin index of -1.
    - Bin edges are calculated using quantiles to ensure approximately equal distribution of values across bins.
    - The rightmost bin includes values exactly equal to range_max.
    """
    # Filter array to include only values within the specified range
    filtered_values = array[(array >= range_min) & (array <= range_max)]

    # Calculate the bin edges using quantiles
    bin_edges = np.quantile(filtered_values, np.linspace(0, 1, n_bins + 1))

    def assign_bin(value):
        # Check if the value is NaN
        if np.isnan(value):
            return -1

        # If the value is outside the range, return it as is
        if value < range_min or value > range_max:
            return value

        # Assign bin based on which range the value falls into
        # We use right=True to ensure that values exactly equal to range_max are included in the last bin
        return np.digitize(value, bin_edges, right=True)

    return assign_bin


# a dictionary that maps the column names to the corresponding mapping function
# - For naturally categorical, we map the NaN values to -1
# - For numerical, we check the codebook for all possible values
#   and map the range of numerical values to a range of bins (e.g. 0-30 to 4 bins)
#   then map the rest to their own category (don't know, didn't want to answer, Nan etc.)
# We only keep vaguely relevant columns to our problem. We will further refine this list later.

mapping_dict = {
    "GENHLTH": lambda value: value if not np.isnan(value) else -1,
    "PHYSHLTH": to_categorical(
        array=x_train[:, features == "PHYSHLTH"].flatten(),
        range_min=0,
        range_max=30,
        n_bins=4,
    ),
    "MENTHLTH": to_categorical(
        array=x_train[:, features == "MENTHLTH"].flatten(),
        range_min=0,
        range_max=30,
        n_bins=4,
    ),
    "POORHLTH": to_categorical(
        array=x_train[:, features == "POORHLTH"].flatten(),
        range_min=0,
        range_max=30,
        n_bins=4,
    ),
    "HLTHPLN1": lambda value: value if not np.isnan(value) else -1,
    "MEDCOST": lambda value: value if not np.isnan(value) else -1,
    "CHECKUP1": lambda value: value if not np.isnan(value) else -1,
    "BPHIGH4": lambda value: value if not np.isnan(value) else -1,
    "BPMEDS": lambda value: value if not np.isnan(value) else -1,
    "BLOODCHO": lambda value: value if not np.isnan(value) else -1,
    "CHOLCHK": lambda value: value if not np.isnan(value) else -1,
    # "CVDINFR4": lambda value: 1 if value == 1 else 0,
    # "CVDCRHD4": lambda value: 1 if value == 1 else 0,
    "TOLDHI2": lambda value: value if not np.isnan(value) else -1,
    "CVDSTRK3": lambda value: value if not np.isnan(value) else -1,
    "ASTHMA3": lambda value: value if not np.isnan(value) else -1,
    "ASTHNOW": lambda value: value if not np.isnan(value) else -1,
    "CHCSCNCR": lambda value: value if not np.isnan(value) else -1,
    "CHCOCNCR": lambda value: value if not np.isnan(value) else -1,
    "CHCCOPD1": lambda value: value if not np.isnan(value) else -1,
    "HAVARTH3": lambda value: value if not np.isnan(value) else -1,
    "ADDEPEV2": lambda value: value if not np.isnan(value) else -1,
    "CHCKIDNY": lambda value: value if not np.isnan(value) else -1,
    "DIABETE3": lambda value: value if not np.isnan(value) else -1,
    "SEX": lambda value: value if not np.isnan(value) else -1,
    "MARITAL": lambda value: value if not np.isnan(value) else -1,
    "EDUCA": lambda value: value if not np.isnan(value) else -1,
    "VETERAN3": lambda value: value if not np.isnan(value) else -1,
    "INCOME2": lambda value: value if not np.isnan(value) else -1,
    "INTERNET": lambda value: value if not np.isnan(value) else -1,
    "WTKG3": to_categorical(
        array=x_train[:, features == "WTKG3"].flatten(),
        range_min=23,
        range_max=295,
        n_bins=6,
    ),
    "QLACTLM2": lambda value: value if not np.isnan(value) else -1,
    "USEEQUIP": lambda value: value if not np.isnan(value) else -1,
    "BLIND": lambda value: value if not np.isnan(value) else -1,
    "DECIDE": lambda value: value if not np.isnan(value) else -1,
    "DIFFWALK": lambda value: value if not np.isnan(value) else -1,
    "DIFFDRES": lambda value: value if not np.isnan(value) else -1,
    "DIFFALON": lambda value: value if not np.isnan(value) else -1,
    "SMOKE100": lambda value: value if not np.isnan(value) else -1,
    "SMOKDAY2": lambda value: value if not np.isnan(value) else -1,
    "LASTSMK2": lambda value: value if not np.isnan(value) else -1,
    "USENOW3": lambda value: value if not np.isnan(value) else -1,
    "AVEDRNK2": to_categorical(
        array=x_train[:, features == "AVEDRNK2"].flatten(),
        range_min=1,
        range_max=76,
        n_bins=5,
    ),
    "DRNK3GE5": to_categorical(
        array=x_train[:, features == "DRNK3GE5"].flatten(),
        range_min=1,
        range_max=76,
        n_bins=5,
    ),
    "EXERANY2": lambda value: value if not np.isnan(value) else -1,
    # "EXERHMM1": lambda value: str(value//200) if value <= 959 and value not in [777,999] else -1,
    "LMTJOIN3": lambda value: value if not np.isnan(value) else -1,
    "FLUSHOT6": lambda value: value if not np.isnan(value) else -1,
    "PDIABTST": lambda value: value if not np.isnan(value) else -1,
    "PREDIAB1": lambda value: value if not np.isnan(value) else -1,
    "INSULIN": lambda value: value if not np.isnan(value) else -1,
    "CIMEMLOS": lambda value: value if not np.isnan(value) else -1,
    "_RFHLTH": lambda value: value if not np.isnan(value) else -1,
    "_HCVU651": lambda value: value if not np.isnan(value) else -1,
    "_RFHYPE5": lambda value: value if not np.isnan(value) else -1,
    "_CHOLCHK": lambda value: value if not np.isnan(value) else -1,
    "_RFCHOL": lambda value: value if not np.isnan(value) else -1,
    # "_MICHD": lambda value: value if value <= 2 else -1,
    "_LTASTH1": lambda value: value if not np.isnan(value) else -1,
    "_CASTHM1": lambda value: value if not np.isnan(value) else -1,
    "_DRDXAR1": lambda value: value if not np.isnan(value) else -1,
    "_AGEG5YR": lambda value: value if not np.isnan(value) else -1,
    "_AGE_G": lambda value: value if not np.isnan(value) else -1,
    "HTM4": to_categorical(
        array=x_train[:, features == "HTM4"].flatten(),
        range_min=0.91,
        range_max=2.44,
        n_bins=6,
    ),
    "_RFBMI5": lambda value: value if not np.isnan(value) else -1,
    "_EDUCAG": lambda value: value if not np.isnan(value) else -1,
    "_SMOKER3": lambda value: value if not np.isnan(value) else -1,
    "_RFBING5": lambda value: value if not np.isnan(value) else -1,
    "_BMI5CAT": lambda value: value if not np.isnan(value) else -1,
    "_RFDRHV5": lambda value: value if not np.isnan(value) else -1,
    "FTJUDA1_": to_categorical(
        array=x_train[:, features == "FTJUDA1_"].flatten(),
        range_min=0,
        range_max=99.99,
        n_bins=4,
    ),
    "MAXVO2_": to_categorical(
        array=x_train[:, features == "MAXVO2_"].flatten(),
        range_min=0,
        range_max=50.1,
        n_bins=6,
    ),
    "ACTIN11_": lambda value: value if not np.isnan(value) else -1,
    "ACTIN21_": lambda value: value if not np.isnan(value) else -1,
    "_PACAT1": lambda value: value if not np.isnan(value) else -1,
    "_PA150R2": lambda value: value if not np.isnan(value) else -1,
    "_PA300R2": lambda value: value if not np.isnan(value) else -1,
    "_PASTRNG": lambda value: value if not np.isnan(value) else -1,
    "_PASTAE1": lambda value: value if not np.isnan(value) else -1,
    "_LMTACT1": lambda value: value if not np.isnan(value) else -1,
    "_LMTWRK1": lambda value: value if not np.isnan(value) else -1,
    "_LMTSCL1": lambda value: value if not np.isnan(value) else -1,
    "_INCOMG": lambda value: value if not np.isnan(value) else -1,
}


# Not necessary because we will do a greedy approach later (unless we want to save time)
def select_features_with_low_nan_ratio(X, features_to_check, threshold=0.1):
    """
    Select features from the training data that have a NaN ratio below a specified threshold.

    Parameters:
    X (numpy.ndarray): The feature data array.
    features_to_check (list): List of features to check for NaN ratios.
    threshold (float): The maximum allowable NaN ratio for a feature to be selected. Default is 0.1.

    Returns:
    list: A list of features that have a NaN ratio below the specified threshold.
    """
    nan_ratios = {}
    for feature in features_to_check:
        nan_ratios[feature] = np.sum(np.isnan(X[:, features == feature])) / len(X)

    selected_features = [
        feature for feature in nan_ratios if nan_ratios[feature] < threshold
    ]

    return selected_features


# This is the function that applies the mapping to the selected features
def apply_mapping(X, selected_features, mapping_dict):
    """
    Applies a mapping function to selected features in the training data.

    Parameters:
    X (numpy.ndarray): The feature data array of shape (n_samples, n_features).
    selected_features (list): A list of feature indices to which the mapping functions will be applied (they need to be keys of mapping_dict)
    mapping_dict (dict): A dictionary where keys are feature indices and values are functions that map feature values.

    Returns:
    numpy.ndarray: A new array with the same number of samples as X but only the selected features,
                   with the mapping functions applied to each feature.
    """
    X_filtered = np.zeros((X.shape[0], len(selected_features)))
    for feature in selected_features:
        feature_values = X[:, features == feature].flatten()
        if feature_values.size > 0:
            X_filtered[:, selected_features.index(feature)] = np.array(
                [mapping_dict[feature](value) for value in feature_values]
            )
    return X_filtered


def fix_class_imbalance(X, y, target_value=1, dont_balance=False):
    """
    Fix class imbalance by oversampling the minority class or undersampling the majority class.

    Parameters:
    X (numpy.ndarray): Feature matrix of shape (n_samples, n_features)
    y (numpy.ndarray): Target vector of shape (n_samples,), containing values -1 and 1
    target_value (int): Class value to balance to (default is 1)
    dont_balance (bool): If True, the function will not balance the classes (default is False)

    Returns:
    X_balanced (numpy.ndarray): Feature matrix with balanced classes
    y_balanced (numpy.ndarray): Balanced target vector
    """
    if dont_balance:
        return X, y

    # Separate samples by class
    class_1_indices = np.where(y == target_value)[0]
    class_minus_1_indices = np.where(y != target_value)[0]

    # Find class counts
    class_1_count = len(class_1_indices)
    class_minus_1_count = len(class_minus_1_indices)

    if class_1_count == class_minus_1_count:
        # If classes are already balanced, return the original data
        return X, y

    elif class_1_count < class_minus_1_count:
        # If class 1 is the minority, oversample class 1
        oversample_size = class_minus_1_count - class_1_count
        oversampled_indices = np.random.choice(
            class_1_indices, oversample_size, replace=True
        )
        new_indices = np.concatenate([np.arange(len(y)), oversampled_indices])
    else:
        # If class -1 is the minority, oversample class -1
        oversample_size = class_1_count - class_minus_1_count
        oversampled_indices = np.random.choice(
            class_minus_1_indices, oversample_size, replace=True
        )
        new_indices = np.concatenate([np.arange(len(y)), oversampled_indices])

    # Create the balanced dataset
    X_balanced = X[new_indices]
    y_balanced = y[new_indices]

    return X_balanced, y_balanced


def one_hot_encode(X, selected_features):
    """
    One-hot encodes the selected features in the input matrix X.

    Parameters:
    X (ndarray): The input feature matrix of shape (n_samples, n_features).
    selected_features (list): A list of selected feature indices for one-hot encoding.

    Returns:
    ndarray: A matrix containing the one-hot encoded columns.
    """
    # Initialize a list to collect the one-hot encoded columns
    one_hot_encoded_list = []

    # Iterate over the final selected features
    for i, feature in enumerate(selected_features):
        # Extract feature values for the current feature
        feature_values = X[:, i]

        # Get unique values for this feature
        unique_values = np.unique(feature_values)

        # One-hot encode by creating a column for each unique value and append to the list
        one_hot_encoded_columns = [
            (feature_values == value).astype(int).reshape(-1, 1)
            for value in unique_values
        ]
        one_hot_encoded_list.extend(one_hot_encoded_columns)

    # Concatenate all one-hot encoded columns horizontally
    X_OHE = np.hstack(one_hot_encoded_list)

    return X_OHE


def label_encode(X, selected_features):
    """
    Label encodes the selected features in the input matrix X.

    Parameters:
    X (ndarray): The input feature matrix of shape (n_samples, n_features).
    selected_features (list): A list of selected feature indices for label encoding.

    Returns:
    ndarray: A matrix containing the label encoded columns.
    """
    # Initialize a list to collect the label encoded columns
    label_encoded_list = []

    # Iterate over the final selected features
    for i, feature in enumerate(selected_features):
        # Extract feature values for the current feature
        feature_values = X[:, i]

        # Get unique values for this feature
        unique_values = np.unique(feature_values)

        # Label encode by mapping each unique value to an integer from 0 to n_categories - 1
        label_encoded_column = np.array(
            [np.where(unique_values == value)[0][0] for value in feature_values]
        ).reshape(-1, 1)
        label_encoded_list.append(label_encoded_column)

    # Concatenate all label encoded columns horizontally
    X_LE = np.hstack(label_encoded_list)

    return X_LE


def map_y_to_0_1(y):
    """
    Maps the target values from -1 and 1 to 0 and 1.

    Parameters:
    y (ndarray): The target vector containing values -1 and 1.

    Returns:
    ndarray: The target vector with values 0 and 1.
    """
    return (y + 1) // 2


def map_y_to_minus_1_1(y):
    """
    Maps the target values from 0 and 1 to -1 and 1.

    Parameters:
    y (ndarray): The target vector containing values 0 and 1.

    Returns:
    ndarray: The target vector with values -1 and 1.
    """
    return 2 * y - 1


# Function to calculate the point-biserial correlation between a feature (x) and a binary target (y)
def point_biserial_correlation(x, y):
    # Calculate the mean of the feature values for y == 1
    y_mean_1 = np.mean(x[y == 1])

    # Calculate the mean of the feature values for y == -1
    y_mean_0 = np.mean(x[y == -1])

    # Calculate the standard deviation of the feature values
    y_std = np.std(x)

    # Calculate the proportion of samples in each class
    p = np.sum(y == 1) / len(y)
    q = 1 - p

    # Calculate the point-biserial correlation using the formula
    correlation = (y_mean_1 - y_mean_0) * np.sqrt(p * q) / y_std if y_std > 0 else 0

    return correlation


# Function to calculate the point-biserial correlation for each selected feature in the dataset
def calculate_correlations_point_biserial(x_train, y_train, selected_features):
    correlations = {}

    # Iterate through each feature and calculate its correlation with the target variable
    for idx, feature_name in enumerate(selected_features):
        # Extract the feature values for the current feature
        feature_values = x_train[:, idx]

        # Calculate the point-biserial correlation between the feature and target
        correlation = point_biserial_correlation(feature_values, y_train)

        # Store the correlation in the dictionary with the feature name as the key
        correlations[feature_name] = correlation

    # Print all correlations as a list
    correlation_list = [
        (feature, correlations[feature]) for feature in selected_features
    ]
    print("\nList of features and their correlations:")
    for feature, corr in correlation_list:
        print(f"{feature}: {corr:.2f}")

    return correlations


# Run the code with your provided dataset and mappings
all_features = list(mapping_dict.keys())
x_train_filtered2 = apply_mapping(x_train, all_features, mapping_dict)

# Extract only the target variable from y_train to create a 1D array
if y_train.ndim > 1 and y_train.shape[1] > 1:
    y_train_target = y_train[:, 1]
else:
    y_train_target = y_train

# Ensure y_train_target is a 1D array with the correct length
y_train_target = y_train_target.flatten()

# Convert x_train_filtered to numpy array if needed
x_train_filtered_array = np.array(x_train_filtered2)

# Now run the point-biserial correlation calculations
correlations = calculate_correlations_point_biserial(
    x_train_filtered_array, y_train_target, all_features
)

# Only select the features with more than 0.1 for absolute value of correlation
threshold = 0.05
selected_features_biserial = [
    feature for feature, corr in correlations.items() if abs(corr) >= threshold
]
print(f"Selected {len(selected_features_biserial)} features out of {len(all_features)}")
print(selected_features_biserial)


def split_data_k_folds(x, y, n_folds=4):
    """
    Splits the dataset into k folds for cross-validation.
    Parameters:
    x (numpy.ndarray): The input features of the dataset.
    y (numpy.ndarray): The target labels of the dataset.
    n_folds (int, optional): The number of folds to split the data into. Default is 5.
    Returns:
    list of tuples: A list where each tuple contains four elements:
        - x_train (numpy.ndarray): Training set features for the current fold.
        - y_train (numpy.ndarray): Training set labels for the current fold.
        - x_test (numpy.ndarray): Test set features for the current fold.
        - y_test (numpy.ndarray): Test set labels for the current fold.
    """
    # Shuffle the data
    indices = np.random.permutation(x.shape[0])

    # Split indices into n equal-sized parts
    fold_sizes = np.full(
        n_folds, x.shape[0] // n_folds, dtype=int
    )  # Base size of each fold
    fold_sizes[: x.shape[0] % n_folds] += 1  # Distribute the remainder

    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]  # Select current fold as test set
        train_indices = np.concatenate(
            [indices[:start], indices[stop:]]
        )  # Rest are training

        x_train, y_train = x[train_indices], y[train_indices]
        x_test, y_test = x[test_indices], y[test_indices]
        folds.append((x_train, y_train, x_test, y_test))

        current = stop

    return folds


class LogisticRegression:
    def __init__(self, max_iters=300, gamma=0.2, lambda_=0.001):
        self.max_iters = max_iters
        self.gamma = gamma
        self.lambda_ = lambda_
        self.w = None

    def fit(self, X, y):
        self.w, _ = reg_logistic_regression(
            y, X, self.lambda_, np.zeros(X.shape[1]), self.max_iters, self.gamma
        )

    def predict(self, X):
        return np.array([1 if p > 0.5 else 0 for p in sigmoid(X @ self.w)])


def accuracy_precision_recall_f1(y_true, y_pred):
    """
    Calculate accuracy, precision, recall, and F1 score for binary classification (y in {0,1}, 1 being positive).

    Parameters:
    y_true (array-like): True binary labels.
    y_pred (array-like): Predicted binary labels.

    Returns:
    tuple: A tuple containing:
        - accuracy (float): The accuracy of the predictions.
        - precision (float): The precision of the predictions.
        - recall (float): The recall of the predictions.
        - f1 (float): The F1 score of the predictions.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return accuracy, precision, recall, f1


def cleaning_x_pipeline(
    x_train,
    y_train,
    x_test,
    features,
    nan_threshold=0.1,
    n_folds=4,
    dont_balance=False,
    oneHot=False,
):
    np.random.seed(41)

    selected_features = select_features_with_low_nan_ratio(
        x_train, features, threshold=nan_threshold
    )
    print(
        "Features with low NaN ratio selected. Number of features:",
        len(selected_features),
    )

    # cleaning
    x_train_filtered_mapped = apply_mapping(x_train, selected_features, mapping_dict)
    x_test_filtered_mapped = apply_mapping(x_test, selected_features, mapping_dict)
    print("Features mapped to categorical values.")

    if oneHot:
        # One-hot encoding
        combined = np.vstack((x_train_filtered_mapped, x_test_filtered_mapped))
        combined_encoded = one_hot_encode(combined, selected_features)
        x_train_encoded = combined_encoded[: len(x_train_filtered_mapped)]
        x_test_encoded = combined_encoded[len(x_train_filtered_mapped) :]
        print("Features one-hot encoded.")
    else:
        # Label encoding
        # we need to combine the training and test set so that the encoding is the same for both
        combined = np.vstack((x_train_filtered_mapped, x_test_filtered_mapped))
        combined_encoded = label_encode(combined, selected_features)
        x_train_encoded = combined_encoded[: len(x_train_filtered_mapped)]
        x_test_encoded = combined_encoded[len(x_train_filtered_mapped) :]
        print("Features label encoded.")

    # no splitting (for submission)
    if n_folds == 0:
        # fix class imbalance in the training set
        x_train_encoded_fixed, y_train_fixed = fix_class_imbalance(
            x_train_encoded, y_train, target_value=1, dont_balance=dont_balance
        )
        print("Class imbalance fixed.")

        return x_train_encoded, x_train_encoded_fixed, y_train_fixed, x_test_encoded
    # splitting (for cross-validation)
    else:
        # split the data into k folds
        folds = split_data_k_folds(x_train_encoded, y_train, n_folds=n_folds)
        print("Data split into k folds.")
        balanced_folds = []

        for x_train_fold, y_train_fold, x_test_fold, y_test_fold in folds:
            # fix class imbalance in the training set
            x_train_fold_fixed, y_train_fold_fixed = fix_class_imbalance(
                x_train_fold, y_train_fold, target_value=1, dont_balance=dont_balance
            )
            balanced_folds.append(
                (
                    x_train_fold,
                    x_train_fold_fixed,
                    y_train_fold,
                    y_train_fold_fixed,
                    x_test_fold,
                    y_test_fold,
                )
            )

        print("Class imbalance fixed.")

        return balanced_folds, x_test_encoded


# Adjusted `fit_predict_model()` function
def fit_predict_model(
    x_train,
    y_train,
    x_test,
    final_features,
    dont_balance=False,
    model=LogisticRegression(),
    oneHot=True,
):
    y_train_mapped = map_y_to_0_1(y_train[:, 1])
    pipeline_output = cleaning_x_pipeline(
        x_train,
        y_train_mapped,
        x_test,
        final_features,
        nan_threshold=0.1,
        n_folds=0,
        dont_balance=dont_balance,
        oneHot=oneHot,
    )

    if len(pipeline_output) == 2:
        # Scenario with cross-validation (k-folds)
        folds, x_test_encoded = pipeline_output
        all_y_preds = []

        for (
            x_train_fold,
            x_train_fold_fixed,
            y_train_fold,
            y_train_fold_fixed,
            x_test_fold,
            y_test_fold,
        ) in folds:
            model.fit(x_train_fold_fixed, y_train_fold_fixed)
            y_pred_fold = model.predict(x_test_fold)
            y_train_pred_fold = model.predict(x_train_fold)

            # Store metrics for each fold if needed
            accuracy, precision, recall, f1 = accuracy_precision_recall_f1(
                y_train_fold, y_train_pred_fold
            )
            print(
                f"Fold training set: accuracy={accuracy:.2f}, precision={precision:.2f}, recall={recall:.2f}, F1={f1:.5f}"
            )

            # Append all predictions for concatenation
            all_y_preds.extend(y_pred_fold)

        # Convert list of predictions to a numpy array
        y_pred = np.array(all_y_preds)
    else:
        # Scenario with no cross-validation (submission mode)
        x_train_encoded, x_train_encoded_fixed, y_train_fixed, x_test_encoded = (
            pipeline_output
        )
        model.fit(x_train_encoded_fixed, y_train_fixed)
        y_pred = model.predict(x_test_encoded)
        y_train_pred = model.predict(x_train_encoded)

        # Print metrics for the training set
        accuracy, precision, recall, f1 = accuracy_precision_recall_f1(
            y_train_mapped, y_train_pred
        )
        print(
            f"Training set: accuracy={accuracy:.2f}, precision={precision:.2f}, recall={recall:.2f}, F1={f1:.5f}"
        )

    return y_pred


final_features = selected_features_biserial
y_pred_test = fit_predict_model(
    x_train,
    y_train,
    x_test,
    final_features,
    dont_balance=False,
    model=LogisticRegression(),
    oneHot=True,
)

Ids = x_test[:, 0]
y_pred_test_final = map_y_to_minus_1_1(y_pred_test)

np.savetxt(
    "final_submission.csv",
    np.array([Ids, y_pred_test_final]).T,
    delimiter=",",
    fmt="%d",
    header="Id,Prediction",
    comments="",
)
