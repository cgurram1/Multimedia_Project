import numpy as np
def svd(data_matrix, k):
    # print(data_matrix.shape)
    cov_matrix = np.dot(data_matrix.T, data_matrix)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    eigenvectors_k = eigenvectors[:, :k]
    singular_values = np.sqrt(eigenvalues[:k])
    U_reduced = np.real(np.dot(data_matrix, eigenvectors_k))
    VT_reduced = np.real(eigenvectors_k.T)
    S_reduced = np.real(np.diag(singular_values))
    latent_semantics = np.dot(U_reduced, S_reduced)
    # print(latent_semantics.shape)
    return U_reduced, S_reduced, VT_reduced

