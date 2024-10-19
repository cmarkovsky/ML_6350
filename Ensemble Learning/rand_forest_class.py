import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count

class TreeNode:

    def __init__(self, value = None, split_feature = None, categories = [], children = []):
        self.value = value
        self.split_feature = split_feature
        self.categories = categories
        self.children = children


    def add_child(self, child):
        self.children.append(child)
    
    def get_child(self, category):
        try:
            return self.children[np.where(self.categories == category)[0][0]]
        except:
            print(f'Category {category} not found in {self.categories}')
            # child = TreeNode(value = Counter(self.value).most_common()[0][0])
            # return TreeNode(value = 0)
 
class DecisionTree:
    def __init__(self, n_features = -1):
        self.root = None
        self.n_features = n_features
        self.categories = None
        self.indxs_dict = None
        self.pred = None
        self.votes = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y)
    
    def _build_tree(self, X, y):        
        feature_index, feature, max_gain = self._determine_ig(X, y)

        split_indxs = self._split_data(X, feature)
        new_children = []
        # print(max_gain)
        if max_gain == 0:
            return TreeNode(value = Counter(y).most_common()[0][0])
        
        for indx in split_indxs:
            if len(indx) > 0:
                
                new_child = self._build_tree(X.iloc[list(indx)].reset_index(drop = True), 
                                             y.iloc[list(indx)].reset_index(drop = True))
                new_children.append(new_child)
        return TreeNode(split_feature=feature, categories=X[feature].unique(), children= new_children)
    
    def _split_data(self, X, feature):
            split_categories = X[feature].unique() 
            indxs = [np.where(X[feature] == category)[0] for category in split_categories]
            return indxs
    
    def _determine_ig(self, X, y):
        m, n = X.shape

        max_gain = float(-1)
        n_feature_indxs = np.arange(n)
        if self.n_features > 0:
            n_feature_indxs = np.random.randint(0, n, self.n_features)
            n = self.n_features
        # print(f'Features: {X.columns[n_feature_indxs]}')
        for i in range(n):
            indxs_dict = {}

            feature = X.columns[n_feature_indxs[i]]
            
            categories = np.unique(X.values[:, i])
            full_entropy = self._entropy(y)
            cat_entropies = []
            for cat in categories:
                indxs = X[X.iloc[:, i] == cat].index
                indxs_dict[cat] = indxs
                category_entropy = (len(indxs) / len(y) * self._entropy(y.loc[indxs]))
                cat_entropies.append(category_entropy)


            gain = full_entropy - np.sum(cat_entropies)
            if gain > max_gain:
                max_gain = gain
                feature_index = i
                split_feature = feature
        # print(f'Feature: {split_feature}, Gain: {gain}')
        return feature_index, split_feature, max_gain
            
    def _entropy(self, y):
        es = []
        for label in y.unique():
            es.append(-1* (len(y.loc[y == label]) / len(y)) * np.log2(len(y.loc[y == label]) / len(y)))
        return sum(es)
    
    def print_tree(self):
        self._print_tree_recursive(self.root)

    def _print_tree_recursive(self, node, indent = ''):
        if node.value is not None:
            # class_probabilities = [f"{label}: {count}" for label, count in node.value.items()]
            print(f"{indent}LEAF: {node.value}")
            return
        feature_name = node.split_feature
        print(f"{indent}{feature_name}")

        for i, category in enumerate(node.categories):
            print(f"{indent}├── == {category}:")
            self._print_tree_recursive(node.children[i], indent + "│   ")

    def predict(self, X):
        y_pred = []
        if isinstance(X, pd.DataFrame):
            for _, row in X.iterrows():
                pred = self._predict_sample(row, self.root)
                y_pred.append(pred)
        elif isinstance(X, pd.Series) or isinstance(X, np.ndarray):
            y_pred.append(self._predict_sample(X, self.root))
        else:
            raise ValueError('X must be a pandas DataFrame, Series, or numpy array')
        
        return y_pred

    def _predict_sample(self, sample, node):
        
        if node.value is not None:
            return node.value
        sample_category = sample[node.split_feature]
        if sample_category in node.categories:
            child = node.get_child(sample_category)
            return self._predict_sample(sample, child)
        else:
            # print(f'Category {sample_category} not found in {node.categories}')
            values = []
            
            for child in node.children:
                    if child.value is not None:
                        values.append(child.value)
            try:            
                vote = Counter(values).most_common()[0][0]
            except:
                vote = -1
            # print(f'Voting: {vote}')
            return vote

class RandomForest:

    def __init__(self, n_features = 2, n_trees=10):
        self.n_trees = n_trees
        self.n_features = n_features
        self.trees = []
    
    def fit(self, X, y, frac = 0.02):
        def _fit_tree(i):
            print(f'Fitting tree {i+1}')
            X_sample = X.sample(frac=frac, replace=True)
            y_sample = y.iloc[X_sample.index]
            
            tree = DecisionTree(n_features=self.n_features)
            tree.fit(X_sample, y_sample)
            return tree
        
        trees = Parallel(n_jobs=cpu_count())(delayed(_fit_tree)(i) for i in range(self.n_trees))
        self.trees = trees
        
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        y_pred = np.sign(np.sum(tree_preds, axis=0))
        return y_pred
    
    def plot_errors(self, X, y):
        def compute_error(i):
            partial_trees = self.trees[:i]
            tree_preds = np.array([tree.predict(X) for tree in partial_trees])
            y_pred = np.sign(np.sum(tree_preds, axis=0))
            error = np.sum(y_pred != y)
            return error / len(y)

        errors = Parallel(n_jobs=cpu_count())(delayed(compute_error)(i) for i in range(1, self.n_trees + 1))

        plt.plot(range(1, self.n_trees + 1), errors, marker='o')
        plt.xlabel('Number of Trees')
        plt.ylabel('Error')
        plt.title('Error with each iteration')
        plt.show()

def compare_feature_set_sizes(X_train, y_train, X_test, y_test, n_trees=5):
    feature_set_sizes = [2, 4, 6]
    errors_dict_train = {}
    errors_dict_test = {}

    for n_features in feature_set_sizes:
        print(f'Fitting Random Forest with {n_features} features')
        rf = RandomForest(n_features=n_features, n_trees=n_trees)
        rf.fit(X_train, y_train)
        
        def compute_error(i, X, y):
            partial_trees = rf.trees[:i]
            tree_preds = np.array([tree.predict(X) for tree in partial_trees])
            y_pred = np.sign(np.sum(tree_preds, axis=0))
            error = np.sum(y_pred != y)
            return error / len(y)

        errors_train = Parallel(n_jobs=-1)(delayed(compute_error)(i, X_train, y_train) for i in range(1, rf.n_trees + 1))
        print('Training Error Computed')
        errors_test = Parallel(n_jobs=-1)(delayed(compute_error)(i, X_test, y_test) for i in range(1, rf.n_trees + 1))
        print('Test Error Computed')

        errors_dict_train[n_features] = errors_train
        errors_dict_test[n_features] = errors_test

    fig, axs = plt.subplots(1, len(feature_set_sizes), figsize=(15, 5), sharey=True)
    for ax, (n_features, errors_train) in zip(axs, errors_dict_train.items()):
        errors_test = errors_dict_test[n_features]
        ax.plot(range(1, rf.n_trees + 1), errors_train, marker='', label=f'Train Error ({n_features} features)')
        ax.plot(range(1, rf.n_trees + 1), errors_test, marker='', label=f'Test Error ({n_features} features)')
        ax.set_xlabel('Number of Trees')
        ax.set_ylabel('Error')
        ax.set_title(f'Error for {n_features} features')
        ax.legend()
    plt.tight_layout()
    plt.show()


# path = './data/bank/train.csv'
# dp = DataProcessor(path)
# X_train, y_train = dp.X_train, dp.y_train
# path = './data/bank/test.csv'
# dp = DataProcessor(path)
# X_test, y_test = dp.X_train, dp.y_train

# compare_feature_set_sizes(X_train, y_train, X_test, y_test, n_trees=10)
# rf = RandomForest(n_features = 4, n_trees=5)
# rf.fit(X_train, y_train)
# # y_pred = rf.predict(X_train)
# rf.plot_errors(X_train, y_train)

