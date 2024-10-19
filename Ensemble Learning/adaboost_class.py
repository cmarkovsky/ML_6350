import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class StumpNode:

    def __init__(self, value = None, split_feature = None, vote = None, indxs = None, children = []):
        self.value = value
        self.split_feature = split_feature
        self.children = children
        self.indxs = indxs
        self.vote = vote

    def add_child(self, child):
        self.children.append(child)
    
    def get_child(self, value):
        for child in self.children:
            if child.value == value:
                return child
        return None

class DecisionStump:
    def __init__(self, sample_weights = None):
        self.feature_index = None
        self.feature = None
        self.root = None
        self.max_gain = None
        self.categories = None
        self.indxs_dict = None
        self.sample_weights = sample_weights
        self.min_error = None
        self.pred = None
        self.votes = None

    def fit(self, X, y):
        self._determine_ig(X, y)            
        self.root = self._build_tree(X, y)
  
    def _predict_sample(self, X):

        category = X.values[self.feature_index]
        child = self.root.get_child(category)
        if child is None:
            print('No child')
            raise ValueError('No child')
        else:
            return child.vote
            
    def predict(self, X):
        y_pred = []
        if isinstance(X, pd.DataFrame):
            for _, row in X.iterrows():
                pred = self._predict_sample(row)
                y_pred.append(pred)
        elif isinstance(X, pd.Series) or isinstance(X, np.ndarray):
            y_pred.append(self._predict_sample(X))
        else:
            raise ValueError('X must be a pandas DataFrame, Series, or numpy array')
        
        return y_pred
    
    def _build_tree(self, X, y):        
        root = StumpNode()
        
        root.split_feature = self.feature
        for i in range(len(self.categories)):
            value = self.categories[i]
            vote = self.votes[i]
            child = StumpNode(value=value, split_feature=self.feature, vote = vote)
            root.add_child(child)
        return root
    
    def _determine_weighted_error(self, X, y):
        m, n = X.shape
        min_error = float('inf')
        for i in range(n):
            indxs_dict = {}
            feature = X.columns[i]
            categories = np.unique(X.values[:, i])
            votes = []
            cat_errors = []
            # print(feature)
            for cat in categories:
                indxs = X[X.iloc[:, i] == cat].index
                vote = Counter(y.iloc[indxs]).most_common(1)[0][0]
                votes.append(vote)
                cat_error = sum(self.sample_weights[indxs] * (vote != y.iloc[indxs]))
                # print(f'Category: {cat}, Error: {cat_error}')
                cat_errors.append(cat_error)

            
            error = sum(cat_errors)    
   
            if error < min_error:
                min_error = error
                self.feature_index = i
                self.feature = feature
                self.categories = categories
                self.votes = votes
                self.indxs = indxs
                self.min_error = error
        return self.feature_index, self.feature
    
    def _determine_ig(self, X, y):
        m, n = X.shape

        max_gain = float(-1)

        #Runs through all features
        for i in range(n):
            indxs_dict = {}
            feature = X.columns[i]
            categories = np.unique(X.values[:, i])
            full_entropy = self._entropy(y, self.sample_weights)
            cat_entropies = []
            votes = []
            for cat in categories:
                indxs = X[X.iloc[:, i] == cat].index
                indxs_dict[cat] = indxs
                cat_weights = self.sample_weights[indxs]
                vote = Counter(y.iloc[indxs]).most_common(1)[0][0]
                votes.append(vote)
                category_entropy = (sum(cat_weights) * self._entropy(y.iloc[indxs], weights=self.sample_weights))
                cat_entropies.append(category_entropy)

            gain = full_entropy - np.sum(cat_entropies)
            # print(f'Feature: {feature}, Gain: {gain}')
            if gain > max_gain:
                max_gain = gain
                self.feature_index = i
                self.feature = feature
                self.categories = categories
                self.max_gain = gain
                self.indxs_dict = indxs_dict
                self.votes = votes
        
        return self.feature_index, self.feature
            
    def _entropy(self, y, weights):
        es = []
        for label in y.unique():
            w = np.sum(weights[y[y == label].index])
            es.append(-1* (w) * np.log2(w))
        return sum(es)

class AdaBoost:

    def __init__(self, n_classifiers=50):
        self.n_classifiers = n_classifiers
        self.alphas = []
        self.trees = []
        self.errors = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.ones(n_samples) / n_samples
        y_preds = []
        for i in range(self.n_classifiers):
            print(f"Iteration {i + 1}")
            
            ds = DecisionStump(sample_weights=w)
            
            ds.fit(X, y)
            self.trees.append(ds)
            y_pred = ds.predict(X)
            y_preds.append(y_pred)

            error = np.sum(w * (y_pred != y))
            if error == 0:
                error += 1e-10
            self.errors.append(error)
            # print(f'Error: {error}')
            alpha = 0.5 * np.log((1 - error) / (error))
            self.alphas.append(alpha)
            w *= np.exp(-alpha * y * y_pred)
            w /= np.sum(w)
            w = np.array(w)
    
    def _ada_predict(self, X):
        y_preds = []
        for i in range(self.n_classifiers):
            y_pred = self.trees[i].predict(X)
            y_preds.append(y_pred)

        fin = np.zeros(len(y_preds[0]))
        for i in range(self.n_classifiers):
            y_preds[i] = np.array(y_preds[i])
            y_preds[i] = [self.alphas[i] * y_preds[i][j] for j in range(len(y_preds[i]))]
            fin += y_preds[i]
        fin_pred = [np.sign(fin[j]) for j in range(len(fin))]

        return fin_pred
    
    def compute_error(self, X, y):
        y_pred = self._ada_predict(X)
        return np.sum(y_pred != y) / len(y)
    
    
    def plot_stump_errors(self, X_train, y_train, X_test, y_test):
        train_errors = []
        test_errors = []
        for i in range(self.n_classifiers):
            tree = self.trees[i]
            pred_train = tree.predict(X_train)
            train_error = np.sum(pred_train != y_train) / len(y_train)
            pred_test = tree.predict(X_test)
            test_error = np.sum(pred_test != y_test) / len(y_test)
            train_errors.append(train_error)
            test_errors.append(test_error)
            
        plt.plot(range(1, self.n_classifiers + 1), train_errors, label='Training Error')
        plt.plot(range(1, self.n_classifiers + 1), test_errors, label='Test Error')
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title('Training and Test Error vs Iterations')
        plt.legend()
        plt.show()

    def plot_full_error(self, X_train, y_train, X_test, y_test):
        train_errors = []
        test_errors = []
        for i in range(1, self.n_classifiers + 1):
            y_train_pred = self._ada_predict(X_train)
            train_error = np.sum(y_train_pred != y_train) / len(y_train)
            train_errors.append(train_error)
            
            y_test_pred = self._ada_predict(X_test)
            test_error = np.sum(y_test_pred != y_test) / len(y_test)
            test_errors.append(test_error)
        
        plt.plot(range(1, self.n_classifiers + 1), train_errors, label='Training Error')
        plt.plot(range(1, self.n_classifiers + 1), test_errors, label='Test Error')
        plt.xlabel('Iterations')
        plt.ylabel('Error')
        plt.title('Training and Test Error vs Iterations')
        plt.legend()
        plt.show()
