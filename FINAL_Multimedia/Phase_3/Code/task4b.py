import task4a
import numpy as np
import pymongo
import torch
import torchvision.datasets as datasets
from PIL import Image
import resnet_feature_extraction
import printing_images_dict
from helpers import fetch_constant, os_path_separator

def hash_vector(query_vector, random_vectors):
    return tuple((np.dot(query_vector, rv) > 0) for rv in random_vectors)

def query(query_vector, hash_tables, random_vectors):
    all_candidates = []
    
    for layer_idx in range(len(hash_tables)):
        hash_key = hash_vector(query_vector, random_vectors[layer_idx])
        converted_tuple = tuple(arr.item() for arr in hash_key)
        if converted_tuple in hash_tables[layer_idx]:
            all_candidates.extend(hash_tables[layer_idx][converted_tuple])
            
    unique_candidates_set = set(all_candidates)
    num_overall_images = len(all_candidates)
    num_unique_images = len(unique_candidates_set)
    unique_candidates_list = list(unique_candidates_set)
    
    return num_overall_images, num_unique_images, unique_candidates_list


def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def image_search(query_vector, data_matrix, hash_tables, random_vectors, t):
    num_overall_images, num_unique_images, candidate_indices = query(query_vector, hash_tables, random_vectors)
    distances = [euclidean_distance(query_vector, data_matrix[idx, :]) for idx in candidate_indices]
    sorted_indices = np.argsort(distances)
    top_t_similar_ids = [candidate_indices[idx] * 2 for idx in sorted_indices[:t]]
    return num_overall_images, num_unique_images, top_t_similar_ids


def task4b():  
    cl = pymongo.MongoClient(fetch_constant('mongo_db_cs'))
    db = cl["caltech101db"]
    collection = db[fetch_constant('phase2Trainingset_collection_name')]

    caltech101_directory = fetch_constant("caltech_dataset_path")
    dataset = datasets.Caltech101(caltech101_directory, download=False)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)
    
    query_image_id = None
    query_image_file = None
    
    print("Enter '1' if you want to give an image ID as input or enter '2' if you want to give Image File as an Input: ")
    query_type = int(input("Enter query type: "))
    
    if query_type == 1:
        query_image_id = int(input("Enter query image ID: "))
    elif query_type == 2:
        query_image_file = input("Give the query image file path: ")
    else: 
        print("Enter valid query type!")
    query_image_data=None
    if query_image_id != None:
        for image_id, (image, label) in enumerate(dataset):
            if image_id == int(query_image_id):
                query_image_data = image
                break
    elif query_image_file != None:
        query_image_data = Image.open(query_image_file)
    
    layer3_feature_descriptor, avgpool_feature_descriptor, fc_feature_descriptor = resnet_feature_extraction.resnet_features(query_image_data)
    reshaped_fc = np.array(fc_feature_descriptor)[np.newaxis, :]
    query_vector = list(reshaped_fc)
    
        
    hash_tables, random_vectors = task4a.task4a()
    data_matrix = np.loadtxt(fetch_constant("data_matrix_path") + os_path_separator() + "resnet50_fc.csv", 
                             delimiter=',')
    # data = task4a.svd(data_matrix, 256)
    t = int(input("Enter the value of t: "))
    num_overall_images, num_unique_images, top_t_similar_ids = image_search(query_vector, data_matrix, hash_tables, random_vectors, t)
    print(f"Total {num_overall_images} images are considered during the process and {num_unique_images} are unique images among them\n")
    images_to_display = {image_id: {'image': image, 'score': 1} for image_id, (image, label) in enumerate(dataset) if image_id in top_t_similar_ids}
    np.savetxt("4bresults.csv", top_t_similar_ids, delimiter=',')
    printing_images_dict.print_images(images_to_display, f"Top {t} similar images", target_size=(224, 224))
    

if __name__ == "__main__":
    task4b()