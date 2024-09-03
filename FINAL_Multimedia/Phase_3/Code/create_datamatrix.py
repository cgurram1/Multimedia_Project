import pymongo 
import numpy as np
import torchvision.datasets as datasets
import os
import sys
from helpers import fetch_constant,os_path_separator

def createMatrix(feature_descriptor):
    data_vector = collection.find_one({"image_id": 0, })[feature_descriptor]
    data_vector = np.array(data_vector).flatten()
    ncolumns = len(data_vector)
    data_matrix = np.empty((0, ncolumns))
    for i in range(8678):
        if i%2 == 0:
            data_vector = collection.find_one({"image_id": i, })[feature_descriptor]
            data_vector = np.array(data_vector).flatten()
            data_matrix = np.vstack((data_matrix, data_vector))
        if(i%200==0):
            print(i)
    return data_matrix

if __name__ == "__main__":
    cl = pymongo.MongoClient(fetch_constant('mongo_db_cs'))
    db = cl["caltech101db"]
    collection = db[fetch_constant('phase2Trainingset_collection_name')]
    caltech101_directory = fetch_constant('caltech_dataset_path')
    dataset = datasets.Caltech101(caltech101_directory, download=False)
    
    feature_models = ["color_moments", "hog", "resnet50_layer3", "resnet50_avgpool", "resnet50_fc","resnet_softmax"]
    for i in range(len(feature_models)):
        data_matrix = createMatrix(feature_models[i] + "_feature_descriptor")
        file_path = fetch_constant('data_matrix_path')+os_path_separator()+f"{feature_models[i]}.csv"
        np.savetxt(file_path, data_matrix, delimiter=",")