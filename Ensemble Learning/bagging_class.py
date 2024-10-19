import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, cpu_count

class DataProcessor:
    def __init__(self, path):
        self.median_dict = {}
        self.features = None
        self.labels = None
        self.path = path
        self.X, self.y = self._process_data(self.path)

    def _process_data(self, path):
        col = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign',
            'pdays','previous', 'poutcome', 'y']
        self.features = col[:-1]
        self.labels = ['no', 'yes']
        df = pd.read_csv(path, names  = col)
        X = df.drop(columns=['y'])
        X = self._conv_numerics(X)

        y = df['y']
        y = self._convert_labels(y)
        
        return X, y

    def _conv_numerics(self, X):
            df = pd.DataFrame()
            for feature in self.features:
                numeric = type(X[feature].iloc[0]) != str
                if numeric:
                    df[feature] = (np.median(X[feature]) < X[feature])
                    self.median_dict[feature] = np.median(X[feature])
                else:
                    df[feature] = X[feature]
            return df

    def _convert_labels(self, y):
        return pd.Series(np.where(y == self.labels[1], 1, -1))

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
    def __init__(self):
        self.root = None
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
        
        for i in range(n):
            indxs_dict = {}
            feature = X.columns[i]
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

class BaggedDecisionTrees:

    def __init__(self, n_trees=100):
        self.n_trees = n_trees
        self.trees = []
    
    def fit(self, X, y, frac = .02):
        def _fit_tree(i):
            X_sample = X.sample(frac=frac, replace=True)
            # print(X_sample)
            y_sample = y.iloc[X_sample.index]
            
            print(f'Fitting Tree {i + 1}')
            tree = DecisionTree()
            tree.fit(X_sample, y_sample)
            return tree
        
        trees = Parallel(n_jobs=cpu_count())(delayed(_fit_tree)(i) for i in range(self.n_trees))
        self.trees = trees
    
    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        y_pred = np.sign(np.sum(tree_preds, axis=0))
        return y_pred
    
    def plot_errors(self, X_train, y_train, X_test, y_test):
        def compute_error(i, X, y):
            partial_trees = self.trees[:i]
            tree_preds = np.array([tree.predict(X) for tree in partial_trees])
            y_pred = np.sign(np.sum(tree_preds, axis=0))
            error = np.sum(y_pred != y)
            return error / len(y)

        train_errors = Parallel(n_jobs=-1)(delayed(compute_error)(i, X_train, y_train) for i in range(1, self.n_trees + 1))
        print('Training Error Computed')

        test_errors = Parallel(n_jobs=-1)(delayed(compute_error)(i, X_test, y_test) for i in range(1, self.n_trees + 1))
        print('Test Error Computed')

        plt.plot(range(1, self.n_trees + 1), train_errors, marker='', label='Train Error')
        plt.plot(range(1, self.n_trees + 1), test_errors, marker='', label='Test Error')
        plt.xlabel('Number of Trees')
        plt.ylabel('Error')
        plt.title('Train vs Test Error with each iteration')
        plt.legend()
        plt.show()