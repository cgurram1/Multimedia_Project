import numpy as np
from PIL import Image
from helpers import fetch_constant
import pymongo
from extracting_feature_space import hog
from extracting_feature_space import resnet_features
from extracting_feature_space import color_moments
import torchvision.datasets

cl = pymongo.MongoClient(fetch_constant('mongo_db_cs'))
db = cl["caltech101db"]
odd_images_collection = db[fetch_constant('odd_images_collection_name')]
dataset = torchvision.datasets.Caltech101(root = fetch_constant("caltech_dataset_path"),download = True)
# Function to extract and store feature descriptors in a MongoDB collection
def feature_descriptors_extraction(collection, dataset):
    # Iterate over each image in the dataset
    for image_id, (image, label) in enumerate(dataset): 
        doc_arr=[] 
        if(image_id%100==0):
            print(image_id)
        if image_id % 2 != 0:
        # Check if the image is grayscale and convert it to RGB if necessary
            if(len(np.array(image).shape) != 3):
                converted_img  = np.stack((np.array(image),) * 3, axis=-1)
                image = Image.fromarray(converted_img)
                
            # Extract color moments, HOG, and ResNet features for the image
            color_moments_feature_descriptor = color_moments.color_moments(image)
            hog_feature_descriptor = hog.HOG(image)
            avgpool_feature_descriptor, layer3_feature_descriptor, fc_feature_descriptor = resnet_features.resnet_features(image)    
            softmax = resnet_features.resnetSoftmax(fc_feature_descriptor)
            # Insert the feature descriptors and related information into the MongoDB collection
            doc_arr.append({
                "image_id": image_id,
                "label": label,
                "color_moments_feature_descriptor": color_moments_feature_descriptor, 
                "hog_feature_descriptor": hog_feature_descriptor,
                "resnet50_layer3_feature_descriptor": layer3_feature_descriptor,
                "resnet50_avgpool_feature_descriptor": avgpool_feature_descriptor,
                "resnet50_fc_feature_descriptor": fc_feature_descriptor,
                "resnet_softmax_feature_descriptor": softmax
            })
            if(len(doc_arr)==100):
                collection.insert_many(doc_arr)
                doc_arr==[]
            # collection.insert_one({
            #     "image_id": image_id,
            #     "label": label,
            #     "color_moments_feature_descriptor": color_moments_feature_descriptor, 
            #     "hog_feature_descriptor": hog_feature_descriptor,
            #     "resnet50_layer3_feature_descriptor": layer3_feature_descriptor,
            #     "resnet50_avgpool_feature_descriptor": avgpool_feature_descriptor,
            #     "resnet50_fc_feature_descriptor": fc_feature_descriptor,
            #     "resnet_softmax_feature_descriptor": softmax
            # })
        if(len(doc_arr)!=0):
            collection.insert_many(doc_arr)
            doc_arr==[]

feature_descriptors_extraction(odd_images_collection,dataset)
