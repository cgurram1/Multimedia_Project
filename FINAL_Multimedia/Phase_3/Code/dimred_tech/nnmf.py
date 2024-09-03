import numpy as np
def nnmf(datamatrix,k):
    n_topics=k
    X=datamatrix
    W = np.random.rand(X.shape[0], n_topics)
    H = np.random.rand(n_topics, X.shape[1])
    for i in range(100):  # number of iterations
        H *= (W.T @ X) / (W.T @ W @ H + 1e-10)
        W *= (X @ H.T) / (W @ H @ H.T + 1e-10)
    return W,H.T
