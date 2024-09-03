from helpers import fetch_constant, os_path_separator
from dimred_tech import svd
from inherent_dim import find_inherent_dim
import torchvision.datasets as datasets
import numpy as np
import pymongo


cl = pymongo.MongoClient(fetch_constant('mongo_db_cs'))
db = cl["caltech101db"]
collection = db[fetch_constant('phase2Trainingset_collection_name')]
featurespace="resnet50_fc"
for i in range(101):
    
    records = list(collection.find({'label': i},{f"{featurespace}_feature_descriptor":1}))
    datamatrix=[i[f'{featurespace}_feature_descriptor'] for i in records]
    datamatrix=np.array(datamatrix)
    inherent_dim=find_inherent_dim(datamatrix,svd.svd)

    for j in range(len(inherent_dim)):
        print(f"label No: {i} Image No: {j} inherent dimenionality: {inherent_dim[j]}")
    



