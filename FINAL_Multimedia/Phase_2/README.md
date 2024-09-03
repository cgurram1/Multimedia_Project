# Multimedia Feature Extraction and Image Retrieval

This repository encompasses an array of code snippets and functions designed for multimedia feature extraction and image retrieval tasks. It provides capabilities for feature extraction, including color moments, HOG (Histogram of Oriented Gradients), and the utilization of a pre-trained ResNet-50 model from ImageNet. The repository is structured to encompass distinct modules dedicated to each of the 11 specified tasks in Phase 2. Furthermore, it offers scripts for data processing, matrix creation, and the generation of database collections to facilitate task-specific operations. Additionally, a task for identifying the top K most similar images within a database, based on a provided input image, is included.

## Pre-requisites 
1.	Create Data Matrices: Open the VSCode terminal or your preferred terminal application. Navigate to the directory containing create_data_matrix.py and execute the python script using the following command: python create_data_matrix.py. It creates all the data matrices (4339 * 900) for color_moments and hog, (4339 * 1024) and (4339 * 1000) for resnet50 in “data_matrices” folder.
2.	Create Representative Image Database: Open the VSCode terminal or your preferred terminal application. Navigate to the directory containing create_rep_images.py and execute the python script using the following command: python create_rep_images.py. It creates a collection with name “labelrepresentativeimages” in the database which contains the representative images for all the labels in all feature spaces.
3.	Create Label Label Similarity matrices: Open the VSCode terminal or your preferred terminal application. Navigate to the directory containing create_label_label_sm.py and execute the python script using the following command: python create_label_label_sm.py. It creates label-label similarity matrices for all the feature spaces in the “label_label_sm_matrices” folder.
4.	Create Image Image Similarity matrices: Open the VSCode terminal or your preferred terminal application. Navigate to the directory containing create_image_image_sm.py and execute the python script using the following command: python create_image_image_sm.py. It creates image-image similarity matrices for all the feature spaces in the “image_image_sm_matrices” folder.
5.	Create the output folder: Create a directory called “outputs” for storing the latent semantics and weight pairs.

## Code Structure

The code is organized into different modules, each serving a specific purpose:

### `task0a.py`

This module will extract and store the features from Caltech101 Dataset and stores in MongoDB along with original labels

### `task0b.py`

This module when provided Image Id or Image File Location, K and feature space will provide K closest images along scores

### `task1.py`

This module when provided query label, K and feature space will provide K closest labels along scores

### `task2a.py`

This module when provided Image Id or Image File Location, K and feature space will provide K closest labels along scores

### `task2b.py`

This module when provided Image Id or Image File Location, K will provide K closest labels along scores using RESNET50 neutral network.

### `task3.py` LS1

This module when given feature space, K and Dimensionality Reduction Technique (SVD,NNMF,LDA,K-Means) will provide top-k extracted latent features and stores in output file and provides ImageID - weightPairs in decreasing order

### `task4.py` LS2

This module when feature space, K will provide top-k extracted latent features using CP-Decompostion of a three Modal (image-feature-label) tensor and stores in output file and provides ImageID - weightPairs in decreasing order

### `task5.py` LS3

This module when given feature space, K and Dimensionality Reduction Technique (SVD,NNMF,LDA,K-Means)
will create label-label similarity matrix and perform dimensionality reduction based on used requested technique will provide top-k extracted latent features and stores in output file and provides ImageID - weightPairs in decreasing order

### `task6.py` LS4

This module when given feature space, K and Dimensionality Reduction Technique (SVD,NNMF,LDA,K-Means)
will create image-image similarity matrix and perform dimensionality reduction based on used requested technique will provide top-k extracted latent features and stores in output file and provides ImageID - weightPairs in decreasing order

### `task7.py`

This Module when given K, Latent Space (LS1,LS2,LS3,LS4) and query image will provide the k most similar images along with their scores

### `task8.py`

This Module when given K, Latent Space (LS1,LS2,LS3,LS4) and query image will provide the k most similar labels along with their scores

### `task9.py`

This Module when given K, Latent Space (LS1,LS2,LS3,LS4) and query label will provide the k most similar labels along with their scores

### `task10.py`

This Module when given K, Latent Space (LS1,LS2,LS3,LS4) and query label will provide the k most similar images along with their scores

### `task11.py`

This module when given Latent Space (LS1, LS2, LS3, LS4), n, and query label as an input will provide m similar images relevant to the label.

## Usage

To use these functions and modules, you can follow these steps:

1. Import the required modules in your Python script or Jupyter Notebook.

2. In the directory where the code folder is present, add the data folder to same directory.

3. Open the code folder and run `create_data_matrix.py`, `create_label_label_sm.py` & `create_image_image_sm` to generate the data matrices.

4. Go to the tasks directory in code directory where you will find the scripts for each task. 

2. Run the `task{number}.py` to start the execution of the desired task

3. Customize the input and parameters as needed for your specific use case.

4. Run your script to perform the desired multimedia tasks.

## Example Usage

Color Moments data matrix, label matrix & image matrix are already generated. Use those matrices to perform any task using any latent semantics.

## Dependencies

The code relies on the following Python libraries:
- `numpy`
- `torch` (PyTorch)
- `torchvision`
- `PIL` (Python Imaging Library)
- `matplotlib`
- `pymongo` (for database operation)
- `Tensorly`

Ensure that you have these libraries installed in your Python environment.

## License

Feel free to customize and use it for your multimedia and image retrieval projects!
