import pymongo
import torchvision.datasets as datasets
import numpy as np
import torch
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
sys.path.append(os.path.join(os.getcwd(), "../../code"))
path = os.getcwd()
import extracting_feature_space.color_moments as color_moments
import extracting_feature_space.hog as hog
import extracting_feature_space.resnet_features as resnet_features
import dimensionality_reduction_techniques.svd as svd
import dimensionality_reduction_techniques.nnmf as nnmf
import dimensionality_reduction_techniques.lda as lda
import dimensionality_reduction_techniques.kmeans as kmeans
import dimensionality_reduction_techniques.cp as cp
import phase_1.printing_images_dict as printing_images_dict
import tasks.task3 as task3
import tasks.task4 as task4
import tasks.task5 as task5
import tasks.task6 as task6

def task7_execution(query_image_data, query_latent_semantics, K, dataset, collection2, query_label):
    
    print("\nSelect a feature model(Select one among these): ")
    print("1. color_moments\n2. hog\n3. resnet50_layer3\n4. resnet50_avgpool\n5. resnet50_fc\n6. resnet50_softmax\n")
    query_feature_model = input("Enter input: ")
    query_feature_model = str(query_feature_model)
    
    k = int(input("Enter k value: "))
    
    if query_latent_semantics != 2:
        print("Enter one of the following dimensional reduction techniques on the chosen feature model:\n")
        print("1. SVD\n2. NNMF\n3. LDA\n4. k-means\n")
        dimredtech = int(input("Enter your choice: "))
        
    if query_label != "":
        query_image_vector = collection2.find({"label": query_label})[0].get(f"{query_feature_model}_rep_image")
        query_image_vector = np.ravel(query_image_vector)
    else:
        if query_feature_model == "color_moments":
                query_image_vector = color_moments.color_moments(query_image_data)
        elif query_feature_model == "hog":
            query_image_vector = hog.HOG(query_image_data)
        else :
            query_layer3_vector, query_avgpool_vector, query_fc_vector = resnet_features.resnet_features(query_image_data)
            if query_feature_model == "resnet50_layer3":
                query_image_vector = query_layer3_vector
            elif query_feature_model == "resnet50_avgpool":
                query_image_vector = query_avgpool_vector
            elif query_feature_model == "resnet50_fc":
                query_image_vector = query_fc_vector 
            else:
                query_image_vector = resnet_features.resnetSoftmax(query_fc_vector)  
        query_image_vector = np.ravel(query_image_vector)
    
    if query_latent_semantics == 1:
        latent_semantics = task3.task3_execution(query_feature_model, k, dimredtech, query_image_vector)
        query_image_vector_ls = latent_semantics[0]
        database_vectors_ls = latent_semantics[1:]
        distances = np.linalg.norm(database_vectors_ls - query_image_vector_ls, axis=1)
        similar_images = {}
        for i, distance in enumerate(distances):
            similar_images[i*2] = distance   
        similar_images = dict(sorted(similar_images.items(), key=lambda x: x[1]))
        top_k_similar_images = dict(list(similar_images.items())[:K])
        images_to_display = {image_id: {'image': image, 'score': top_k_similar_images[image_id]} for image_id, (image, label) in enumerate(dataset) if image_id in top_k_similar_images}
        printing_images_dict.print_images(images_to_display, "LS" + str(query_latent_semantics), target_size=(224, 224))
        
    if query_latent_semantics == 2:
        latent_semantics = task4.task4_execution(query_feature_model, k, query_image_vector)
        query_image_vector_ls = latent_semantics[0]
        database_vectors_ls = latent_semantics[1:]
        distances = np.linalg.norm(database_vectors_ls - query_image_vector_ls, axis=1)
        similar_images = {}
        for i, distance in enumerate(distances):
            similar_images[i*2] = distance   
        similar_images = dict(sorted(similar_images.items(), key=lambda x: x[1]))
        top_k_similar_images = dict(list(similar_images.items())[:K])
        images_to_display = {image_id: {'image': image, 'score': top_k_similar_images[image_id]} for image_id, (image, label) in enumerate(dataset) if image_id in top_k_similar_images}
        printing_images_dict.print_images(images_to_display, "LS" + str(query_latent_semantics), target_size=(224, 224))
    
    if query_latent_semantics == 3:
        list_of_rep_images_data = []
        latent_semantics = task5.task5_execution(query_feature_model, k, dimredtech)
        for i in range(101):
            list_of_rep_images_data.append((i, collection2.find({"label": i})[0].get(f"{query_feature_model}_rep_image")))
        most_similar_label = None
        highest_similarity = -1
        for label, image_vector in list_of_rep_images_data:
            similarity = cosine_similarity([query_image_vector], [image_vector])
            if similarity > highest_similarity:
                highest_similarity = similarity
                most_similar_label = label
                
        query_label_vector_ls = latent_semantics[most_similar_label]
        distances = np.linalg.norm(latent_semantics - query_label_vector_ls, axis=1)
        similar_labels = {}
        for i, distance in enumerate(distances):
            similar_labels[i] = distance   
        similar_labels = dict(sorted(similar_labels.items(), key=lambda x: x[1]))
        top_k_similar_labels = dict(list(similar_labels.items())[:K])
        top_k_similar_images = {}
        for label, weight in top_k_similar_labels.items():
            top_k_similar_images[collection2.find({"label": label})[0].get(f"{query_feature_model}_rep_image_id")] = weight
        images_to_display = {image_id: {'image': image, 'score': top_k_similar_images[image_id]} for image_id, (image, label) in enumerate(dataset) if image_id in top_k_similar_images}
        printing_images_dict.print_images(images_to_display, "LS" + str(query_latent_semantics), target_size=(224, 224))
        
    if query_latent_semantics == 4:
        latent_semantics = task6.task6_execution(query_feature_model, k, dimredtech)
        query_image_vector_ls = collection2.find({"label": query_label})[0].get(f"{query_feature_model}_rep_image")
        distances = np.linalg.norm(latent_semantics - query_image_vector_ls, axis=1)
        similar_images = {}
        for i, distance in enumerate(distances):
            similar_images[i*2] = distance   
        similar_images = dict(sorted(similar_images.items(), key=lambda x: x[1]))
        top_k_similar_images = dict(list(similar_images.items())[:K])
        images_to_display = {image_id: {'image': image, 'score': top_k_similar_images[image_id]} for image_id, (image, label) in enumerate(dataset) if image_id in top_k_similar_images}
        printing_images_dict.print_images(images_to_display, "LS" + str(query_latent_semantics), target_size=(224, 224))
    return True

def task7(query_label):
    cl = pymongo.MongoClient("mongodb://localhost:27017")
    db = cl["caltech101db"]
    collection1 = db["phase2trainingdataset"]
    collection1_name = "phase2trainingdataset"
    collection2 = db["labelrepresentativeimages"]
    collection2_name = "labelrepresentativeimages"
    caltech101_directory = os.path.join(path, "../../data")
    dataset = datasets.Caltech101(caltech101_directory, download=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)
    
    query_image_id = None
    query_image_file = None
    
    if query_label == "":
        print("Enter '1' if you want to give an image ID as input or enter '2' if you want to give Image File as an Input: ")
        query_type = int(input("Enter query type: "))
        
        if query_type == 1:
            query_image_id = int(input("Enter query image ID: "))
        elif query_type == 2:
            query_image_file = input("Give the query image file path: ")
        else: 
            print("Enter valid query type!")

        if query_image_id != None:
            for image_id, (image, label) in enumerate(dataset):
                if image_id == int(query_image_id):
                    query_image_data = image
                    break
        elif query_image_file != None:
            query_image_data = Image.open(query_image_file)
        
    print("Enter any of the Latent Semantics: \n1. LS1\n2. LS2\n3. LS3\n4. LS4\n")
    
    query_latent_semantics = int(input("Enter your choice number: "))
    
    K = int(input("Enter K value for finding K similar images: "))
    
    if query_image_id != None:
        for image_id, (image, label) in enumerate(dataset):
            if image_id == int(query_image_id):
                query_image_data = image
                break
    elif query_image_file != None:
        query_image_data = Image.open(query_image_file)
    if query_label == "":    
        task7_execution(query_image_data, query_latent_semantics, K, dataset, collection2, "")
    else:
        task7_execution("", query_latent_semantics, K, dataset, collection2, query_label)
    
if __name__ == "__main__":
    task7("")