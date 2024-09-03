import numpy as np
import pymongo
import torch
import torchvision.datasets as datasets
from helpers import fetch_constant, os_path_separator

def svd(data_matrix, k):
    print(data_matrix.shape)
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
    return latent_semantics

def hash_vector(vector, random_vectors):
    return tuple((np.dot(vector, rv) > 0) for rv in random_vectors)

def index_data(data_matrix, num_layers, num_hashes):
    hash_tables = [{} for _ in range(num_layers)]
    random_vectors = [np.random.randn(num_hashes, data_matrix.shape[1]) for _ in range(num_layers)]

    for layer_idx in range(num_layers):
        for i in range(data_matrix.shape[0]):
            hash_key = hash_vector(data_matrix[i, :], random_vectors[layer_idx])
            if hash_key in hash_tables[layer_idx]:
                hash_tables[layer_idx][hash_key].append(i)
            else:
                hash_tables[layer_idx][hash_key] = [i]

    return hash_tables, random_vectors

def query(query_vector, hash_tables, random_vectors):
    candidates = set()

    for layer_idx in range(len(hash_tables)):
        hash_key = hash_vector(query_vector, random_vectors[layer_idx])
        if hash_key in hash_tables[layer_idx]:
            candidates.update(hash_tables[layer_idx][hash_key])

    return list(candidates)


def task4a():
    cl = pymongo.MongoClient(fetch_constant('mongo_db_cs'))
    db = cl["caltech101db"]
    collection = db[fetch_constant('phase2Trainingset_collection_name')]

    caltech101_directory = fetch_constant("caltech_dataset_path")
    dataset = datasets.Caltech101(caltech101_directory, download=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)
    
    L = int(input("Enter the number of Layers: "))
    H = int(input("Enter the number of hashes per layer: "))
    data_matrix = np.loadtxt(fetch_constant("data_matrix_path") + os_path_separator() + "resnet50_fc.csv"
                             , delimiter=',')
    # data = svd(data_matrix, 256)
    
    hash_tables, random_vectors = index_data(data_matrix, L, H)
    return hash_tables, random_vectors
    

if __name__ == "__main__":
    task4a()