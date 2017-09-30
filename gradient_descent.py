import numpy as np


def sigmoid_func(Y):
    return 1.0 / (1 + np.exp(-Y))


def compute_Gradient(Z, W, Y, step_size):
    inner_Mat = np.identity(len(Z)) + np.dot((1-(2*Z)), Y.T)
    inner_Mat = np.dot(inner_Mat, W)
    return step_size * inner_Mat


def grad_descent(X, U, total_iterations, step_size, isUniform=True):
    lossHistory = []
    if isUniform:
        W = np.random.uniform(0, 0.1, (len(X), len(X)))
    else:
        #W = np.random.random((len(X), len(X)))
        W = np.random.uniform(-0.01, -0.1, (len(X), len(X)))
    #print("W shape ",W.shape)
    for epoch in range(total_iterations):
        print("epoch:",epoch)
        Y = np.dot(W, X)
        Z = sigmoid_func(Y)
        error = Y - U
        lossHistory.append(np.sum(error ** 2))
        W += compute_Gradient(Z, W, Y, step_size)
    Y = np.dot(W, X)
    return lossHistory, Y
