from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import torchvision.datasets as datasets
import torch
import printing_images_dict
from sklearn.naive_bayes import MultinomialNB
from helpers import fetch_constant, os_path_separator

input_image_ids = []
labels = []
caltech101_directory = fetch_constant("caltech_dataset_path")
dataset = datasets.Caltech101(caltech101_directory, download=False)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=8)
def task5():
    print("1 - Very Relevant\n2 - Relevant\n3 - Irrelevant\n4 - Very Irrelevant")
    while True:
        image_id = input("Enter image id (or -1 to stop): ")
        if image_id == '-1':
            break
        label = input("Enter Relevancy Rating for image {}: ".format(image_id))
        input_image_ids.append(image_id)
        labels.append(label)
    user_choice = int(input("1. SVM\n2. Probabilistic Feedback\n"))
    if(user_choice == 1):
        runSVM(input_image_ids,labels)
    elif(user_choice == 2):
        runProbabilisticFeedback(input_image_ids,labels)
    

def runProbabilisticFeedback(image_ids,labels):
    data_matrix = np.loadtxt(fetch_constant("data_matrix_path") + os_path_separator() + "hog.csv", delimiter=',')
    image_vectors = []
    for image_id in image_ids:
        image_id = int(int(image_id)/2)
        image_vector = data_matrix[image_id, :]
        image_vectors.append(image_vector)
    X_train,y_train = image_vectors,labels
    model = build_feedback_model(X_train,y_train)

    image_ids_csv = np.loadtxt("4bresults.csv", delimiter=',', converters={0: converter_func}, dtype=int)
    data_matrix = np.loadtxt(fetch_constant("data_matrix_path") + os_path_separator() + "hog.csv", delimiter=',')
    testing_image_ids = []
    testing_labels = []
    for image_id in image_ids_csv:
        image_id = int(int(image_id)/2)
        row = data_matrix[image_id, :]
        testing_image_ids.append(image_id * 2)
        if(image_id* 2 not in input_image_ids):
            predicted_label = predict_label(model, row)
        else:
            index = input_image_ids.index(image_id * 2)
            predicted_label = labels[index]
        testing_labels.append(predicted_label)
    pairs = list(zip(testing_image_ids, testing_labels))
    sorted_pairs = sorted(pairs, key=lambda x: x[1])
    sorted_testing_image_ids = [pair[0] for pair in sorted_pairs]
    sorted_testing_labels = [pair[1] for pair in sorted_pairs]

        
    images_to_display = {}

    for image_id in sorted_testing_image_ids:
        image, label = dataset[image_id]
        images_to_display[image_id] = {'image': image, 'score': 0}
    
    np.savetxt("task5_reordered_results_Probability.csv", sorted_testing_image_ids, delimiter=',')
    printing_images_dict.print_images(images_to_display, f"Reordered similar images", target_size=(224, 224))
    for i in range(10):
        print("Image ID : " + str(sorted_testing_image_ids[i]) + " , Predicted Label : " + str(sorted_testing_labels[i]))
    
def build_feedback_model(image_features, user_feedback):
    classifier = MultinomialNB()
    classifier.fit(image_features, user_feedback)
    return classifier

def predict_label(model, new_image_vector):
    new_image_vector = np.reshape(new_image_vector, (1, -1))
    predicted_label = model.predict(new_image_vector)[0]

    return predicted_label

def runSVM(image_ids,labels):
    classifier = train_svm_classifier(image_ids,labels)
    image_ids_csv = np.loadtxt("4bresults.csv", delimiter=',', converters={0: converter_func}, dtype=int)
    data_matrix = np.loadtxt(fetch_constant("data_matrix_path") + os_path_separator() + "resnet50_fc.csv", delimiter=',')
    testing_image_ids = []
    testing_labels = []
    for image_id in image_ids_csv:
        image_id = int(int(image_id)/2)
        row = data_matrix[image_id, :]
        testing_image_ids.append(image_id * 2)
        if(image_id* 2 not in input_image_ids):
            predicted_label = predict_label_SVM(classifier,row)
        else:
            index = input_image_ids.index(image_id * 2)
            predicted_label = labels[index]
        testing_labels.append(predicted_label)
    pairs = list(zip(testing_image_ids, testing_labels))
    sorted_pairs = sorted(pairs, key=lambda x: x[1])
    sorted_testing_image_ids = [pair[0] for pair in sorted_pairs]
    sorted_testing_labels = [pair[1] for pair in sorted_pairs]
   
    images_to_display = {}

    for image_id in sorted_testing_image_ids:
        image, label = dataset[image_id]
        images_to_display[image_id] = {'image': image, 'score': 0}
    
    np.savetxt("task5_reordered_results_SVM.csv", sorted_testing_image_ids, delimiter=',')
    printing_images_dict.print_images(images_to_display, f"Reordered similar images", target_size=(224, 224))
    for i in range(10):
        print("Image ID : " + str(sorted_testing_image_ids[i]) + " , Predicted Label : " + str(sorted_testing_labels[i]))


def train_svm_classifier(image_ids, labels):
    data_matrix = np.loadtxt(fetch_constant("data_matrix_path") + os_path_separator() + "resnet50_fc.csv", delimiter=',')
    image_vectors = []
    for image_id in image_ids:
        image_id = int(int(image_id)/2)
        image_vector = data_matrix[image_id, :]
        image_vectors.append(image_vector)
    X_train,y_train = image_vectors,labels
    classifier = svm.SVC(kernel='linear', C=1, decision_function_shape='ovr')
    classifier.fit(X_train, y_train)
    return classifier
def converter_func(value):
    return int(float(value))
def predict_label_SVM(classifier, new_image_vector):
    new_image_vector = np.array(new_image_vector).reshape(1, -1)
    predicted_label = classifier.predict(new_image_vector)
    return predicted_label[0]


if __name__ == "__main__":
    print("execytion started")
    task5()