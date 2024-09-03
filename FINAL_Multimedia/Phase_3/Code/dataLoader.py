from pymongo import MongoClient
import pymongo
import numpy as np
from helpers import fetch_constant, os_path_separator
# Connect to the MongoDB client
client = pymongo.MongoClient(fetch_constant('mongo_db_cs'))

# Select the database and collection
db = client['caltech101db']
collection = db[fetch_constant('phase2Trainingset_collection_name')]

# Query the collection
data_cursor = collection.find({})

# Extract data into a list of dictionaries
data = list(data_cursor)

# Now 'data' contains your dataset
# You can process it as needed, for example, to extract features and labels
def load_feat_data(feature):
    
    features = []
    labels = []

    for item in data:
        # Assuming 'color_moments_feature_descriptor' is one of the features you want to use
        features.append(item[feature])
        # Assuming the label is stored under 'label'
        labels.append(item['label'])

    # Convert to numpy arrays if necessary, for use in machine learning models
    

    features = np.array(features)
    labels = np.array(labels)
    
    return features, labels

def load_test_data(feature):
    client = pymongo.MongoClient(fetch_constant('mongo_db_cs'))
    db = client['caltech101db']
    collection = db['caltech101withGSimages']
    
    # Query to select only documents with odd 'image_id'
    odd_images_cursor = collection.find({"image_id": {"$mod": [2, 1]}}, {feature: 1, "label": 1, "_id": 0})

    # Initialize lists to hold the feature data and labels
    features = []
    labels = []

    # Iterate through the cursor and extract the feature data and labels
    for document in odd_images_cursor:
        features.append(document.get(feature, []))  # Use an empty list as default if the key doesn't exist
        labels.append(document['label'])  # Assuming 'label' key always exists

    return features, labels