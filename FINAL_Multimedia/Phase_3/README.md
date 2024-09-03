# Image Clustering, Indexing & Classification / Relevance Feedback: Multimedia & Web Databases

This repository encompasses an array of code snippets and functions designed for Image Clustering, Indexing & Classification / Relevance Feedback tasks. It provides capabilities for feature extraction and the utilization of a pre-trained ResNet-50 model from ImageNet. The repository is structured to encompass distinct modules dedicated to each of the 5 specified tasks in Phase 3. Additionally, a task for identifying the top K most similar images within a database, based on a provided input image, is included.

## Pre-requisites 
1. Data Matrix: The feature model used in the tasks need to be pre generated from the MongoDB collection as a data matrix of rows containing even images and columns as features
2. The Caltech101 Data folder path is need to be defined in the task files to the directory in which they are present in the local system.
3. Add the directory where Caltech101 data exists in the root directory where the code directory is present. Both the directories need to be present in the same level.

## Code Structure

The code is organized into different modules, each serving a specific purpose:

### `task0a.py`

This module computes and prints the inherent dimensionality associated with the even numbered Caltec101 images.

### `task0b.py`

This module computes and prints the inherent dimensionality associated with each unique label of the even numbered Caltec101 images.

### `task1.py`

This module implements classifier using distance to label specific latent semantics

### `task2.py`

This module implements DBScan algorithm and visualizes the clusters as differently colored point clouds in a 2-dimensional MDS space, and as groups of image thumbnails.

### `task3.py` LS1

This module implements m-NN classifier, decision-tree classifier, PPR based classifier and outputs per-label precision, recall, and F1-score values as well as output an overall accuracy value.

### `task4a.py` LS2

This module implements the locality-sensitive hashing tool for Euclidean distance and create an in-memory index structure containing the given set of vectors.

### `task4a.py` LS2

This module implements an image search algorithm given a query image and t denoting the number of top t images similar images to be displayed and also providing the counts of unique and overall images considered during the process.

### `task5.py` LS3

This module re orders the results from task4b.py using SVM and probabilistic classifiers and visalizes the re ordered images.


## Usage

To use these functions and modules, you can follow these steps:

1. Import the required modules in your Python script or Jupyter Notebook.

2. In the directory where the code folder is present, add the data folder to same directory.

3. Go to the tasks directory in code directory where you will find the scripts for each task. 

4. Run the `task{number}.py` to start the execution of the desired task

3. Customize the input and parameters as needed for your specific use case.

4. Run your script to perform the desired multimedia tasks.

## Example Usage

You can refer to the provided main program `main.py` for an example of how to use these functions to perform multimedia tasks, including classification, indexing and image search.

## Dependencies

The code relies on the following Python libraries:
- `numpy`
- `torch` (PyTorch)
- `torchvision`
- `PIL` (Python Imaging Library)
- `matplotlib`
- `pymongo` (for database operation)
- `Tensorly`
- `Networkx`

Ensure that you have these libraries installed in your Python environment.

## License

Feel free to customize and use it for your multimedia and image retrieval projects!
