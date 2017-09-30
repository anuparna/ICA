import numpy as np


def mixTestData(A, U):
    X = np.dot(A, U)
    return X


def mixData(source_signals):
    rows = source_signals.shape[0]
    A = np.random.random((rows, rows))
    X = np.dot(A, source_signals)
    return X
