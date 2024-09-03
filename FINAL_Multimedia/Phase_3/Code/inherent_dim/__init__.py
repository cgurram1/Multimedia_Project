import numpy as np
def find_inherent_dim(data_matrix,svd_fn):
    k = np.shape(data_matrix)[1]
    U_reduced, S_reduced, VT_reduced = svd_fn(data_matrix, k)
    total_sq_sum = 0
    for i in range(k):
        total_sq_sum += S_reduced[i][i] ** 2

    new_sum = 0
    required_i = None
    for i in range(k):
        new_sum += S_reduced[i][i] ** 2

        if new_sum / total_sq_sum > 0.95:
            required_i = i
            break

    for i in range(required_i + 1, k):
        S_reduced[i][i] = 0
    return np.dot(U_reduced, S_reduced)[:, : required_i + 1]