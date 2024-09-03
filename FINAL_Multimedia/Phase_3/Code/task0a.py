from helpers import fetch_constant, os_path_separator
from dimred_tech import svd
from inherent_dim import find_inherent_dim
import torchvision.datasets as datasets
import numpy as np

caltech101_directory = fetch_constant("caltech_dataset_path")

dataset = datasets.Caltech101(caltech101_directory, download=False)

data_matrix = np.loadtxt(
    fetch_constant("data_matrix_path") + os_path_separator() + "resnet50_fc.csv",
    delimiter=",",
)
inherent_dim=find_inherent_dim(data_matrix,svd.svd)

for i in range(len(inherent_dim)):
    print(f"Image No: {i} inherent dimenionality: {inherent_dim[i]}")
    