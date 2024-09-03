from helpers import fetch_constant
from dimred_tech import svd
from inherent_dim import find_inherent_dim
from accuracy_metrics import accuracy
import torchvision.datasets as datasets
import numpy as np
import pymongo
from sklearn import metrics
import matplotlib.pyplot as plt

#creating MongoDB connection
cl = pymongo.MongoClient(fetch_constant('mongo_db_cs'))
db = cl["caltech101db"]
collection = db[fetch_constant('phase2Trainingset_collection_name')]
rep_collection=db[fetch_constant('representative_collection_name')]

#choosing feature space
featurespace="resnet50_fc"

#K input
k=int(input("Enter the K Value: "))

#fetching the even dataset from MongoDB
even_dataset=list(collection.find({},{f"{featurespace}_feature_descriptor":1,"label":1}))
odd_collection = db[fetch_constant('odd_images_collection_name')]


#fetching the odd dataset from MongoDB
odd_images = list(odd_collection.find({},{f"{featurespace}_feature_descriptor":1,"label":1}))
distance_matrix = []
labels=[]
for i in range(101): 
    records=filter(lambda x: x['label']==i,even_dataset)
    rep_img_id= list(rep_collection.find({'label':i},{f'{featurespace}_rep_image_id':1,f'{featurespace}_rep_image':1}))
    datamatrix=[i[f'{featurespace}_feature_descriptor'] for i in records]
    rep=[i[f'{featurespace}_rep_image'] for i in rep_img_id]
    datamatrix=np.array(datamatrix)

    #Perfrom Latent Space transformation 
    U,S,VT = svd.svd(datamatrix,k)
    rep = (np.array(rep@(VT.transpose())).flatten())
    print(f"Processing {i} class of {101} ",end="\r")
    distances = [] 
    for  item in odd_images:
        img,label =item[f"{featurespace}_feature_descriptor"], item['label']
        odd_image_ls = img@(VT.transpose())

        #computing cosine similarity 
        distances.append(np.dot(odd_image_ls,rep)/(np.linalg.norm(odd_image_ls)*np.linalg.norm(rep)))
    distance_matrix.append(distances)
distance_matrix = np.array(distance_matrix)
predicted_labels = np.argmax(distance_matrix, axis=0)

actual_labels=[i['label'] for i in  odd_images]

accuracy.calculate_labelwise_metrics(accuracy.get_OneHot(actual_labels), accuracy.get_OneHot(predicted_labels))



