import pymongo
import torchvision.datasets as datasets
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../../code"))
path = os.getcwd()
import dimensionality_reduction_techniques.svd as svd
import dimensionality_reduction_techniques.nnmf as nnmf
import dimensionality_reduction_techniques.lda as lda
import dimensionality_reduction_techniques.kmeans as kmeans
import dimensionality_reduction_techniques.cp as cp


def LS4(query_label_rep_image_id, query_feature_model, dimredtech, k, K, list_of_rep_images):
    iism_path = f"{os.path.join(os.getcwd(), f'../image_image_sm_matrices/image_image_sm_{query_feature_model}.csv')}"
    ii_similarity_matrix = np.genfromtxt(iism_path, delimiter=',')

    if dimredtech == 1:
        U_reduced, S_reduced, VT_reduced = svd.svd(ii_similarity_matrix, k)
        latent_space_matrix = np.dot(U_reduced, S_reduced)
    elif dimredtech == 2:
        W, H = nnmf.nmf(ii_similarity_matrix, k)
        latent_space_matrix = W[:, :k] 
    elif dimredtech == 3:
        latent_semantics, query_distribution = lda.lda(ii_similarity_matrix, k)
        latent_space_matrix = query_distribution
    elif dimredtech == 4:
        latent_space_matrix = kmeans.kmeans(ii_similarity_matrix, k)
    else:
        print("Enter valid dimensionality reduction technique choice!!")
        
    query_image_vector_ls = latent_space_matrix[int(query_label_rep_image_id/2)]
    database_vectors_list = []

    for i in list_of_rep_images:
        vector_to_add = latent_space_matrix[int(i/2)]
        database_vectors_list.append(vector_to_add)

    # Convert the list of arrays to a NumPy array
    database_vectors_ls = np.array(database_vectors_list)
    distances = np.linalg.norm(database_vectors_ls - query_image_vector_ls, axis=1)
    similar_labels = {}
    for i, distance in enumerate(distances):
        similar_labels[i] = distance   
    similar_labels = dict(sorted(similar_labels.items(), key=lambda x: x[1]))
    top_k_similar_labels = dict(list(similar_labels.items())[:K])
    for label, weight in top_k_similar_labels.items():
        print("Label : " + str(label), ", weight : " + str(weight))
        

def LS3(query_label_id, query_feature_model, dimredtech, k, K):
    path = f"{os.path.join(os.getcwd(), f'../label_label_sm_matrices/label_label_sm_{query_feature_model}.csv')}"
    ll_similarity_matrix = np.genfromtxt(path, delimiter=",")
    if dimredtech == 1:
        U_reduced, S_reduced, VT_reduced = svd.svd(ll_similarity_matrix, k)
        latent_space_matrix = np.dot(U_reduced, S_reduced)
    elif dimredtech == 2:
        W, H = nnmf.nmf(ll_similarity_matrix, k)
        latent_space_matrix = W[:, :k] 
    elif dimredtech == 3:
        latent_semantics, query_distribution = lda.lda(ll_similarity_matrix, k)
        latent_space_matrix = query_distribution
    elif dimredtech == 4:
        latent_space_matrix = kmeans.kmeans(ll_similarity_matrix, k)
    else:
        print("Enter valid dimensionality reduction technique choice!!")
    query_image_vector_ls = latent_space_matrix[query_label_id]
    # Convert the list of arrays to a NumPy array
    distances = np.linalg.norm(latent_space_matrix - query_image_vector_ls, axis=1)
    similar_labels = {}
    for i, distance in enumerate(distances):
        similar_labels[i] = distance   
    similar_labels = dict(sorted(similar_labels.items(), key=lambda x: x[1]))
    top_k_similar_labels = dict(list(similar_labels.items())[:K])
    for label, weight in top_k_similar_labels.items():
        print("Label : " + str(label), ", weight : " + str(weight))
    
def LS2(query_label_repImg_id, data_matrix, k, K,list_of_rep_image_ds):
    weights,factors = cp.cp(data_matrix, k)
    latent_space_matrix = factors[0]
    query_image_vector_ls = latent_space_matrix[int(query_label_repImg_id/2)]
    database_vectors_list = []
    
    for i in list_of_rep_image_ds:
        vector_to_add = latent_space_matrix[int(i/2)]
        database_vectors_list.append(vector_to_add)
        
    # Convert the list of arrays to a NumPy array
    database_vectors_ls = np.array(database_vectors_list)
    distances = np.linalg.norm(database_vectors_ls - query_image_vector_ls, axis=1)
    similar_labels = {}
    for i, distance in enumerate(distances):
        similar_labels[i] = distance   
    similar_labels = dict(sorted(similar_labels.items(), key=lambda x: x[1]))
    top_k_similar_labels = dict(list(similar_labels.items())[:K])
    for label, weight in top_k_similar_labels.items():
        print("Label : " + str(label), ", weight : " + str(weight))

def LS1(query_label_repImg_id, data_matrix, dimredtech, k, K,list_of_rep_image_ds):
    if dimredtech == 1:
        U_reduced, S_reduced, VT_reduced = svd.svd(data_matrix, k)
        latent_space_matrix = np.dot(U_reduced, S_reduced)
    elif dimredtech == 2:
        W, H = nnmf.nmf(data_matrix, k)
        latent_space_matrix = W[:, :k]
    elif dimredtech == 3:
        latent_semantics, query_distribution = lda.lda(data_matrix, k)
        latent_space_matrix = query_distribution
    elif dimredtech == 4:
        latent_space_matrix = kmeans.kmeans(data_matrix, k)
    else:
        print("Enter valid dimensionality reduction technique choice!!")
        
    print(int(query_label_repImg_id/2))
    print(latent_space_matrix.shape)
    query_image_vector_ls = latent_space_matrix[int(query_label_repImg_id/2)]
    database_vectors_list = []
    for i in list_of_rep_image_ds:
        vector_to_add = latent_space_matrix[int(i/2)]
        database_vectors_list.append(vector_to_add)
    # Convert the list of arrays to a NumPy array
    database_vectors_ls = np.array(database_vectors_list)
    distances = np.linalg.norm(database_vectors_ls - query_image_vector_ls, axis=1)
    similar_labels = {}
    for i, distance in enumerate(distances):
        similar_labels[i] = distance   
    similar_labels = dict(sorted(similar_labels.items(), key=lambda x: x[1]))
    top_k_similar_labels = dict(list(similar_labels.items())[:K])
    for label, weight in top_k_similar_labels.items():
        print("Label : " + str(label), ", weight : " + str(weight))

def task9(query_label_id, query_latent_semantics, K, collection2):
    print("\nSelect a feature model for the latent semantics(Select one among these): ")
    print("1. color_moments\n2. hog\n3. resnet50_layer3\n4. resnet50_avgpool\n5. resnet50_fc\n6. resnet50_softmax\n")
    query_feature_model = input("Enter input ")
    query_feature_model = str(query_feature_model)
    if query_latent_semantics != 2: #LS2 is CP decomposition
        print("Enter one of the following dimensional reduction techniques on the chosen feature model:\n")
        print("1. SVD\n2. NNMF\n3. LDA\n4. k-means\n")
        dimredtech = int(input("Enter your choice number: "))
    k = int(input("Enter k value for dimensionality reduction: "))
    
    list_of_rep_images = []
    data_matrix = np.loadtxt(f"{os.path.join(os.getcwd(), f'../data_matrices/data_matrix_{query_feature_model}.csv')}", delimiter=',')
    query_label_rep_image_id = collection2.find({"label": query_label_id})[0].get(f"{query_feature_model}_rep_image_id")
    for i in range(101):
        list_of_rep_images.append(collection2.find({"label": i})[0].get(f"{query_feature_model}_rep_image_id"))
    print(list_of_rep_images)
    if query_latent_semantics == 1:
        LS1(query_label_rep_image_id, data_matrix, dimredtech, k, K, list_of_rep_images)
    elif query_latent_semantics == 2:
        LS2(query_label_rep_image_id, data_matrix, k, K, list_of_rep_images)
    elif query_latent_semantics == 3:
        LS3(query_label_id, query_feature_model, dimredtech, k, K)
    elif query_latent_semantics == 4:
        LS4(query_label_rep_image_id, query_feature_model, dimredtech, k, K, list_of_rep_images)
    else:
        print("Enter a valid latent semantics choice!!")
    

if __name__ == "__main__":
    cl = pymongo.MongoClient("mongodb://localhost:27017")
    db = cl["caltech101db"]
    collection1 = db["phase2trainingdataset"]
    collection1_name = "phase2trainingdataset"
    collection2 = db["labelrepresentativeimages"]
    collection2_name = "labelrepresentativeimages"

    caltech101_directory = os.path.join(path, "../../data")
    dataset = datasets.Caltech101(caltech101_directory, download=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)
    
    query_label_id = int(input("Enter query Label: "))
        
    print("Enter any of the Latent Semantics: \n1. LS1\n2. LS2\n3. LS3\n4. LS4\n")
    
    query_latent_semantics = int(input("Enter your choice number: "))
    
    K = int(input("Enter K value for finding K similar labels: "))
    
    task9(query_label_id, query_latent_semantics, K, collection2)