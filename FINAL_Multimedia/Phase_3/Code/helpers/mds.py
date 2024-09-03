import numpy as np
import matplotlib.pyplot as plt
from helpers import fetch_constant
import pymongo
cl = pymongo.MongoClient(fetch_constant('mongo_db_cs'))
db = cl["caltech101db"]
collection = db[fetch_constant('phase2Trainingset_collection_name')]
featurespace="resnet50_fc"

def mds(X, n_components):
    n_samples = X.shape[0]

    # Step 1: Compute the pairwise distance matrix
    dist_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            dist_matrix[i, j] = np.linalg.norm(X[i] - X[j])

    # Step 2: Compute the centering matrix
    H = np.eye(n_samples) - np.ones((n_samples, n_samples)) / n_samples

    # Step 3: Perform double centering
    B = -H.dot(dist_matrix**2).dot(H) / 2

    # Step 4: Compute the eigenvalues and eigenvectors of B
    eigvals, eigvecs = np.linalg.eigh(B)

    # Step 5: Select the top n_components eigenvectors and eigenvalues
    idx = np.argsort(eigvals)[::-1][:n_components]
    eigvals, eigvecs = eigvals[idx], eigvecs[:, idx]

    # Step 6: Compute the coordinates of the points in the new space
    L = np.diag(np.sqrt(eigvals))
    V = eigvecs
    Y = V.dot(L)

    return Y

# for k in range(100):
#         label_dataset=list(collection.find({"label":k},{'_id':0,f"{featurespace}_feature_descriptor":1}))
#         print(len(label_dataset))

#         for i in range(len(label_dataset)):
#                 label_dataset[i] = label_dataset[i][f'{featurespace}_feature_descriptor']
#         label_dataset = np.array(label_dataset)
#         reduced_data=mds(label_dataset,2)
#         # print(reduced_data.shape)
        
#         plt.scatter(reduced_data[:,0],reduced_data[:,1],marker=".")
# # plt.scatter(reduced_data[:,0]*2,reduced_data[:,1]*2,marker="+")
# plt.legend()
# plt.show()