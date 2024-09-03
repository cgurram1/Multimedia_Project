import numpy as np
import networkx as nx
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from accuracy_metrics import accuracy

#### helper functions ###

def normalize(X, axis=1):
    norms = np.linalg.norm(X, axis=axis)
    norms[norms == 0] = 1
    return X / norms[:, np.newaxis]


#### CLASSIFIER DEFINITIONS ####

#### m-NN Classifier  ######
class m_NNClassifier:
    def __init__(self, m):
        self.m = m
        self.x_train = None
        self.y_train = None
    
    def fit(self, x_train, y_train, m):
        self.x_train = x_train
        self.y_train = y_train
        self.m= m
        
    def predict(self, query):
        dist_arr = [self.euc_dist(query, train_img) for train_img in self.x_train]
        dist_arr = np.array(dist_arr)
        
        m_neighbour_labels = self.y_train[np.argsort(dist_arr)[:self.m]]
        label_pred = np.argmax(np.bincount(m_neighbour_labels))
        
        return label_pred
        
    def euc_dist(self, x: np.ndarray, y:np.ndarray):
        return np.sqrt(np.sum(np.square(x-y)))
    

#### Decision Tree Classfier  #######
class TreeNode:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None):
        self.feature_index = feature_index  
        self.threshold = threshold  
        self.value = value  
        self.left = left  
        self.right = right  


class DecisionTreeClassifier:
    def __init__(self, max_depth = None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        
    def fit(self, X, y):   
        self.tree = self._build_tree(X, y, depth=0)

    def _calculate_entropy(self, labels):
        _, counts = np.unique(labels, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  
        return entropy
    
    def _calculate_information_gain(self, X, y, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold
        if not left_mask.any() or not right_mask.any():
            return 0

        left_entropy = self._calculate_entropy(y[left_mask])
        right_entropy = self._calculate_entropy(y[right_mask])
        weighted_entropy = (left_entropy * left_mask.sum() + right_entropy * right_mask.sum()) / y.size

        return self._calculate_entropy(y) - weighted_entropy

    def _find_best_split(self, X, y):
        best_info_gain = -1
        best_feature_index = None
        best_threshold = None

        for feature_index in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                info_gain = self._calculate_information_gain(X, y, feature_index, threshold)
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _build_tree(self, X, y, depth):
        if len(set(y)) == 1 or len(y) < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            return TreeNode(value=np.argmax(np.bincount(y)))

        feature_index, threshold = self._find_best_split(X, y)
        if feature_index is None or threshold is None:
            return TreeNode(value=np.argmax(np.bincount(y)))

        left_mask = X[:, feature_index] <= threshold
        right_mask = X[:, feature_index] > threshold

        if not left_mask.any() or not right_mask.any():
            return TreeNode(value=np.argmax(np.bincount(y)))

        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return TreeNode(feature_index=feature_index, threshold=threshold, left=left_subtree, right=right_subtree)


    def predict(self, X):
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, node):
        if node.value is not None:
            return node.value
        else:
            if x[node.feature_index] <= node.threshold:
                return self._predict_tree(x, node.left)
            else:
                return self._predict_tree(x, node.right)
            
            
###### PPR Classifier  ########
class PPRClassifier:
    def __init__(self, alpha=0.85, max_iter=100):
        self.alpha = alpha
        self.max_iter = max_iter
        self.graph = None
        self.features = None
        self.labels = None

    def fit(self, features, labels):
        self.features = features
        self.labels = labels
        self.graph = self.create_graph(features)
        self.pagerank_scores = self.compute_pagerank()

    def create_graph(self, features):
        similarity_matrix = cosine_similarity(features)
        G = nx.Graph()
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                G.add_edge(i, j, weight=similarity_matrix[i][j])
        return G

    def compute_pagerank(self, tol=1.0e-6):
        num_nodes = len(self.graph)
        pagerank_scores = np.full(num_nodes, 1.0 / num_nodes)
        
        adjacency_matrix = nx.to_numpy_array(self.graph)
        row_sums = adjacency_matrix.sum(axis=1)
        transition_matrix = adjacency_matrix / row_sums[:, np.newaxis]

        for iteration in range(self.max_iter):
            new_pagerank_scores = self.alpha * np.matmul(transition_matrix.T, pagerank_scores) + (1 - self.alpha) / num_nodes

            if np.linalg.norm(new_pagerank_scores - pagerank_scores, 1) < tol:
                print(f"PageRank converged after {iteration + 1} iterations.")
                break

            pagerank_scores = new_pagerank_scores

        return {node: pagerank_scores[i] for i, node in enumerate(self.graph.nodes)}


    def get_representatives(self, label, num_representatives):
        label_indices = np.where(self.labels == label)[0]
        label_pagerank_scores = {index: self.pagerank_scores[index] for index in label_indices}
        sorted_indices = sorted(label_pagerank_scores, key=label_pagerank_scores.get, reverse=True)
        return sorted_indices[:min(num_representatives, len(sorted_indices))]

class RepresentativeClassifier:
    def __init__(self, ppr_classifier, num_representatives):
        self.ppr_classifier = ppr_classifier
        self.num_representatives = num_representatives
        self.representatives = {}

    def fit(self, features, labels):
        self.ppr_classifier.fit(features, labels)
        unique_labels = set(labels)

        for label in unique_labels:
            representative_indices = self.ppr_classifier.get_representatives(label, self.num_representatives)
            self.representatives[label] = features[representative_indices]
        print('got representatives')

    def predict(self, test_features):
        predictions = []
        for i, test_feature in enumerate(test_features):
            max_similarity = 0
            predicted_label = None
            
            for label, reps in self.representatives.items():
                similarities = [cosine_similarity([test_feature], [rep]).flatten()[0] for rep in reps]
                avg_similarity = np.mean(similarities)

                if avg_similarity > max_similarity:
                    max_similarity = avg_similarity
                    predicted_label = label
            print(f'predicted for image {i} prediction: {predicted_label}')

            predictions.append(predicted_label)

        return predictions
    
    

### main program ###
def task3():
    from dataLoader import load_feat_data
    print('######Task 3######\n')
    
    print('Loading data...')
    train_feat, train_labs = load_feat_data('color_moments_feature_descriptor')
    train_feat = train_feat.reshape(train_feat.shape[0], -1)
    test_feat, test_labs = load_feat_data('color_moments_feature_descriptor')
    test_feat = test_feat.reshape(test_feat.shape[0], -1)
    print('Data loaded.\n\n')
            
    classifier = int(input('Enter the classifier you want to use:\n1. m-NN classifier\n2. decision tree classifier\n3. Personalized Page Rank classifier\n'))
    match classifier:
        case 1:
            m = int(input('Enter the top "m" matches to consider for classification: '))
            nnclassifier = m_NNClassifier(m)
            nnclassifier.fit(train_feat, train_labs, m)
            predictions = [nnclassifier.predict(query) for query in test_feat]
            print(predictions)
            print(len(predictions))
            # print(metrics.classification_report(test_labs, predictions))
            # print('\n Accuracy value:-', metrics.accuracy_score(test_labs, predictions))
            accuracy.calculate_labelwise_metrics(accuracy.get_OneHot(test_labs), accuracy.get_OneHot(predictions))

        case 2:
            decision_tree = DecisionTreeClassifier(max_depth=10)  
            n_components = 500
            pca = PCA(n_components=n_components)
            train_feat_pca = pca.fit_transform(train_feat)
            test_feat_pca = pca.transform(test_feat)
            decision_tree.fit(train_feat_pca, train_labs)
            predictions = decision_tree.predict(test_feat_pca)
            print(predictions)
            print(len(predictions))
            # print(metrics.classification_report(test_labs, predictions))
            # print('\n Accuracy value:-', metrics.accuracy_score(test_labs, predictions))
            accuracy.calculate_labelwise_metrics(accuracy.get_OneHot(test_labs), accuracy.get_OneHot(predictions))
        case 3:
            
            ppr_classifier = PPRClassifier(alpha=0.5, max_iter=100)
            representative_classifier = RepresentativeClassifier(ppr_classifier, num_representatives=5)
            representative_classifier.fit(train_feat, train_labs)
            predictions = representative_classifier.predict(test_feat)
            
            # print(metrics.classification_report(test_labs, predictions))
            # print('\n Accuracy value:-', metrics.accuracy_score(test_labs, predictions))
            accuracy.calculate_labelwise_metrics(accuracy.get_OneHot(test_labs), accuracy.get_OneHot(predictions))
            

if __name__ == '__main__':
    task3()