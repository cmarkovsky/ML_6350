import numpy as np
import pandas as pd
from collections import Counter

class TreeNode:

    def __init__(self, split_feature = None, feature_categories = None, children = [], value = None):
        self.value = value
        self.split_feature = split_feature
        self.categories = feature_categories
        self.children = children

class Tree:

    def __init__(self, max_depth = 1, gain_metric = 'GI'):
        self.root = None
        self.max_depth = max_depth
        self.features = None
        self.labels = None
        self.gain_metric = gain_metric
        self.median_dict = {}

    

    def train_tree(self, X, y, features, labels):
        X = self._replace_unknown(X, features)
        X = self._conv_numerics(X, features)
        self.features = features
        self.labels = labels
        self.n_labels = len(labels)
        self.root = self._build_tree(X, y, 0)

    def _replace_unknown(self, X, features):
        df = pd.DataFrame()
        for feature in features:
            if 'unknown' in X[feature].unique():
                if Counter(X[feature]).most_common()[0][0] != 'unknown':
                    replace = Counter(X[feature]).most_common()[0][0]
                else:
                    replace = Counter(X[feature]).most_common()[1][0]
                new = X[feature].replace('unknown', replace)
                df[feature] = new
            else:
                df[feature] = X[feature]
        return df
            
        
    def _conv_numerics(self, X, features):
        df = pd.DataFrame()
        
        for feature in features:
            numeric = type(X[feature].iloc[0]) != str
            if numeric:
                df[feature] = (np.median(X[feature]) < X[feature])
                self.median_dict[feature] = np.median(X[feature])
            else:
                df[feature] = X[feature]
        return df
    
    def _build_tree(self, X, y, current_depth):
        if current_depth >= self.max_depth:
            return TreeNode(value = np.unique(y, return_counts=True)[1] / len(y))
        

        split_feature, max_gain, split_categories = self._deter_split(X, y, self.gain_metric)
        
        if split_categories is None:
            leaf = TreeNode(value=np.unique(y, return_counts=True)[1] / len(y))
            return leaf
        
        split_indxs = self._split_data(X[split_feature], split_categories)
        new_children = []

        for indx in split_indxs:
            if len(indx) > 0:
                
                new_child = self._build_tree(X.iloc[list(indx)].reset_index(drop = True), 
                                             y.iloc[list(indx)].reset_index(drop = True), 
                                             current_depth + 1)
                new_children.append(new_child)
        return TreeNode(split_feature=split_feature, feature_categories = split_categories, children= new_children)


    def _split_data(self, X, split_categories):
        indxs = [np.where(X == category)[0] for category in split_categories]
        return indxs

    def _deter_split(self, X, y, method='GI'):
        max_gain = -1
        split_feature = None
        split_categories = None
        for feature in self.features:
            X_input = X[feature]

            gain = self._info_gain(X_input, y, method)
            # print(feature, gain)
            if gain > max_gain:
                max_gain = gain
                split_feature = feature

                split_categories = X[split_feature].unique()
            
        
        return split_feature, max_gain, split_categories
            

    def _info_gain(self, X_feature, y, method  = 'GI'):

        categories = X_feature.unique()
        indxs = [np.where(X_feature == category)[0] for category in categories]

        if method == 'EN':
            full_entropy = self._entropy(y)
            category_entropy = sum(len(indx) / len(y) * self._entropy(y[indx]) for indx in indxs if len(indx) > 0)
            gain = full_entropy - category_entropy
        elif method == 'GI':
            full_GI = self._gini_index(y)
            category_GI = sum(len(indx) / len(y) * self._gini_index(y[indx]) for indx in indxs if len(indx) > 0)
            gain = full_GI - category_GI
        else:
            full_ME = self._majority_error(y)
            category_ME = sum(len(indx) / len(y) * self._majority_error(y[indx]) for indx in indxs if len(indx) > 0)
            gain = full_ME - category_ME

        return gain

    def _gini_index(self, y):
        GIs = []
        for label in y.unique():
            GI = (len(y[y == label]) / len(y)) ** 2
            GIs.append(GI)
        return 1 - sum(GIs)

    def _majority_error(self, y):

        return (len(y) - Counter(y).most_common()[0][1]) / len(y)

    def _entropy(self, y):
        es = []
        for value in y.unique():
            es.append(-1* (len(y[y == value]) / len(y)) * np.log2(len(y[y == value]) / len(y)))
        
        return sum(es)
  
    def predict2(self, X):

            if self.root is None:
                print('Not yet trained')
                return None
            
            predictions = []
            for _, sample in X.iterrows():
                predictions.append(self._predict_sample(sample, self.root))
                
            
            return pd.Series(predictions)

    def _predict_sample(self, sample, node):
        
        if node.value is not None:
            return self.labels[np.argmax(node.value)]
        if node.split_feature in self.median_dict.keys():
            feature_value = sample[node.split_feature] > self.median_dict[node.split_feature]
        else:
            feature_value = sample[node.split_feature]
        for i, category in enumerate(node.categories):
            if feature_value == category:
                return self._predict_sample(sample, node.children[i])