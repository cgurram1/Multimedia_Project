def get_OneHot(lst):
    result = []
    for label in lst:
        temp = [0]*101
        temp[label]= 1
        result.append(temp)
    return result

def calculate_labelwise_metrics(true_labels, predicted_labels):
    num_classes = len(true_labels[0])
    
    label_precisions, label_recalls, label_f1_scores = [], [], []
    
    class_counts = [sum(label[label_idx] for label in true_labels) for label_idx in range(num_classes)]

    true_positive_all, false_positive_all, true_negative_all, false_negative_all = 0, 0, 0, 0

    for label in range(num_classes):
        true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0

        for true_label, predicted_label in zip(true_labels, predicted_labels):
            if true_label[label] == 1 and predicted_label[label] == 1:
                true_positive += 1
            elif true_label[label] == 0 and predicted_label[label] == 1:
                false_positive += 1
            elif true_label[label] == 1 and predicted_label[label] == 0:
                false_negative += 1
            elif true_label[label] == 0 and predicted_label[label] == 0:
                true_negative += 1

        true_positive_all += true_positive
        false_positive_all += false_positive
        true_negative_all += true_negative
        false_negative_all += false_negative

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

        label_precisions.append(precision)
        label_recalls.append(recall)
        label_f1_scores.append(f1_score)

    # Calculate class weights based on the number of samples in each class
    class_weights = [count / sum(class_counts) for count in class_counts]

    # Calculate overall accuracy as the weighted average of per-class accuracies
    overall_accuracy = true_positive_all/len(true_labels)
    print("i am here")
    # print("trial accuracy:",(true_positive_all+false_positive_all)/len(true_label))
    from prettytable import PrettyTable
    my_table = PrettyTable(["Label","Precision","Recall","F1-Score"])
    for label in range(len(label_precisions)):
        my_table.add_row([label,round(label_precisions[label],4),round(label_recalls[label],4),round(label_f1_scores[label],4)])
    print(my_table)
    print("Overall Accuracy: {:.6f}".format(overall_accuracy))
    #return label_precisions, label_recalls, label_f1_scores, overall_accuracy