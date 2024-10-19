import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataProcessor:
    def __init__(self, path):
        self.features = None
        self.labels = None
        self.path = path
        self.X_train, self.y_train = self._process_data(self.path)

    def _process_data(self, path):
        col = ['cement', 'slag', 'fly_ash', 'water', 'sp', 'coarse_agg', 'fine_agg', 'y']
        self.features = col[:-1]
        # self.labels = ['no', 'yes']
        df = pd.read_csv(path, names  = col)
        X = df.drop(columns=['y'])
        # X = self._conv_numerics(X)

        y = df['y']
        # y = self._convert_labels(y)
        
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

class GradientDescent:
     
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-7):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.costs = []
    
    def batch_fit(self, X, y):
        n_samples, n_features = X.shape
        self._init_weights(n_features)
        weights_list = []
        X = np.c_[np.ones((n_samples, 1)), X]

        for i in range(self.max_iter):
            weights_list.append(self.weights)
            y_pred = self._predict(X)
            error = y_pred - y
            grad_w = (1 / n_samples) * np.dot(X.T, error)

            c_weights = list(self.weights)
            self.weights -= self.learning_rate * grad_w

            cost = (1 / (2)) * np.sum(error ** 2)

            self.costs.append(cost)
            x = np.linalg.norm(self.weights - c_weights)
            if np.linalg.norm(self.weights - c_weights) < self.tol:
                print(f'Converged at iteration {i + 1}')
                break
    
    def stochastic_fit(self, X, y):
        n_samples, n_features = X.shape
        self._init_weights(n_features)
        X = np.c_[np.ones((n_samples, 1)), X]
        
        for t in range(self.max_iter):
            indxs = np.arange(n_samples)
            np.random.shuffle(indxs)
            for i in range(n_samples):
                cX = X[indxs[i]]
                cy = y.iloc[indxs[i]]
                y_pred = self._predict(cX)
                error = y_pred - cy
                grad_w = np.dot(cX.T, error)
                c_weights = list(self.weights)
                self.weights -= self.learning_rate * grad_w
            cost = (1 / (2)) * np.sum((self._predict(X) - y) ** 2)
            self.costs.append(cost)
            
            if np.linalg.norm(self.weights - c_weights) < self.tol:
                print(f'Converged at iteration {i + 1}')
                break
    
    def calc_test_cost(self, X, y):
        n_samples, n_features = X.shape
        X = np.c_[np.ones((n_samples, 1)), X]
        cost = (1 / (2)) * np.sum((self._predict(X) - y) ** 2)
        return cost
    
    def _predict(self, X):
        return np.dot(X, self.weights)
    
    def _mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def _init_weights(self, n_features):
        self.weights = np.zeros(n_features + 1)

    def plot_cost(self):
        plt.plot(self.costs)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost vs Iterations')
        plt.show()

