import numpy as np


def compute_loss(y, tx, w):
    # Mean squared error
    e = y - tx.dot(w)
    return 1 / 2 * np.mean(e**2)


def compute_gradient(y, tx, w):
    # Gradient of the mean squared error
    e = y - tx.dot(w)
    return -tx.T.dot(e) / len(e)


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    # Generate a minibatch iterator for a dataset.
    data_size = len(y)
    indices = np.arange(data_size)
    if shuffle:
        np.random.shuffle(indices)
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            batch_indices = indices[start_index:end_index]
            yield y[batch_indices], tx[batch_indices]


# Note that all functions should return: (w, loss), which is the last weight vector of the method, and the corresponding loss value (cost function).
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    # Linear regression using gradient descent
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        # update w by gradient
        w = w - gamma * grad
        print(
            "Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    # Linear regression using stochastic gradient descent
    batch_size = 1
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            # compute a stochastic gradient and loss
            grad = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            # update w by gradient
            w = w - gamma * grad
            print(
                "Stochastic Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
                )
            )
    return w, loss


def least_squares(y, tx):
    # Least squares regression using normal equations
    a = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss


# Moreover, the loss returned by the regularized methods (ridge regression and reg logistic regression) should not include the penalty term
def ridge_regression(y, tx, lambda_):
    # Ridge regression using normal equations
    aI = 2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1])
    a = tx.T.dot(tx) + aI
    b = tx.T.dot(y)
    w = np.linalg.solve(a, b)
    loss = compute_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    # Logistic regression using gradient descent
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        # update w by gradient
        w = w - gamma * grad
        print(
            "Logistic Regression({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # Regularized logistic regression using gradient descent
    w = initial_w
    for n_iter in range(max_iters):
        # compute gradient and loss
        grad = compute_gradient(y, tx, w) + 2 * lambda_ * w
        loss = compute_loss(y, tx, w) + lambda_ * np.linalg.norm(w) ** 2
        # update w by gradient
        w = w - gamma * grad
        print(
            "Regularized Logistic Regression({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )
    return w, loss
