import numpy as np 


# ML Methods 

def compute_loss(y, tx, w):
    """
    Compute the Mean Squared Error (MSE) loss.
    """
    e = y - tx.dot(w)
    return 1/2*np.mean(e**2)

def compute_gradient(y, tx, w):
    """
    Compute the gradient with respect to w.
    This function is used when solving gradient based method.
    """
    e = y - tx.dot(w)
    grad = -(tx.T.dot(e))/ len(e)
    return grad

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index] 

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """
    Perform linear regression using gradient descent with Mean Squared Error (MSE) as the loss function.
    """
    w = initial_w

    for n_iter in range(max_iters):
        # Compute the gradient and the loss
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        # Update the weights using the gradient and learning rate
        w = w - gamma * grad
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi= n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w,max_iters, gamma):
    """
    Perform linear regression using stochastic gradient descent with Mean Squared Error (MSE) as the loss function.
    """
    w = initial_w
    batch_size = 1

    for n_iter in range(max_iters):
        # Generate minibatches using the batch_iter function
        for mini_batch_y, mini_batch_tx in batch_iter(y, tx, batch_size):
            # Compute the gradient and the loss using the mini-batch
            grad = compute_gradient(mini_batch_y, mini_batch_tx, w)
            loss = compute_loss(mini_batch_y, mini_batch_tx, w)
            
            # Update the weight vector
            w = w - gamma * grad
            print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                  bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss

def least_squares(y, tx):
    """
    Solve the least squares problem using the normal equations.
    """
    # Compute the optimal weights using the normal equation
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))

    # Compute the Mean Squared Error (MSE)
    loss = compute_loss(y, tx, w)
    return w, loss

def ridge_regression(y, tx, lambda_):
    """
    Perform Ridge Regression (L2 regularization) using the normal equations.
    """
    # Get the number of features (D) from the tx shape
    D = tx.shape[1]

    # Compute the Ridge Regression solution using the normal equation
    aI = 2 * tx.shape[0]* lambda_ * np.eye(D)
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)

    # Compute the Mean Squared Error (MSE)
    loss = compute_loss(y, tx, w)
    return w, loss

def sigmoid(t):
    """
    Compute the sigmoid function.
    """
    return 1 / (1 + np.exp(-t))

def compute_logistic_loss(y, tx, w):
    """
    Compute the logistic loss (negative log-likelihood).
    """
    pred = sigmoid(tx.dot(w))
    loss = -np.mean(y * np.log(pred) + (1 - y) * np.log(1 - pred))
    return loss

def compute_logistic_gradient(y, tx, w):
    """
    Compute the gradient of the logistic loss.
    """
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y)
    return grad

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent.
    """
    w = initial_w
    for n_iter in range(max_iters):
        # Compute the gradient and loss
        grad = compute_logistic_gradient(y, tx, w)
        loss = compute_logistic_loss(y, tx, w)
        
        # Update the weights
        w = w - gamma * grad
        print("Logistic Regression({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent.
    """
    w = initial_w
        
    for n_iter in range(max_iters):
        # Compute the gradient and loss
        grad = compute_logistic_gradient(y, tx, w) + lambda_ * w  # Add L2 regularization term to gradient
        loss = compute_logistic_loss(y, tx, w) + (lambda_ / 2) * np.sum(w ** 2)  # Add regularization to loss
            
        # Update the weights
        w = w - gamma * grad
        print("Regularized Logistic Regression({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss
        