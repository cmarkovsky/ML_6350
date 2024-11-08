import numpy as np
import matplotlib.pyplot as plt

class Perceptron:

    def __init__(self, n_epochs = 10, lr = 0.01):
        self.lr = lr
        self.epochs = n_epochs
    
    def fit(self, X, y, method = 'averaged'):
        self.n_samples, self.n_features = X.shape
        self._initialize_weights()
        self.weights = []
        self.errors = []
        self.method = method

        if method == 'averaged':
            self._averaged_fit(X, y)
        elif method == 'voted':
            self._voted_fit(X, y)
        elif method == 'standard':
            self._standard_fit(X, y)
        else:
            raise ValueError('Method must be one of the following: averaged, voted, standard')
        return self

    def _voted_fit(self, X, y):
        self.cs = []
        c = 0

        for i in range(self.epochs):
            indices = np.arange(self.n_samples)
            np.random.seed(42)
            np.random.shuffle(indices)
            X = X.iloc[indices]
            y = y.iloc[indices]

            for xi, yi in zip(X.values, y.values):
                y_pred = self._vp_predict_sample(xi)
                target = yi

                if y_pred == target:
                    c += 1
                else:
                    self.weights.append(self.w.copy())
                    self.cs.append(c)

                    self.w[1:] += self.lr * target * xi
                    self.w[0] += self.lr * target
                    c = 1

            full_y_pred = self.predict(X)
            epoch_error = np.mean(full_y_pred != y)
            self.errors.append(epoch_error)
            

            print(f'Epoch: {i+1}, Error: {np.round(epoch_error, 3)}')
            print('-----------------------------------\n')
        self.final_w = zip(self.weights, self.cs)
        return self
    
    def _standard_fit(self, X, y):
        self.n_updates = []

        for i in range(self.epochs):
            n_update = 0

            indices = np.arange(self.n_samples)
            np.random.seed(42)
            np.random.shuffle(indices)
            X = X.iloc[indices]
            y = y.iloc[indices]

            for xi, yi in zip(X.values, y.values):
                
                y_pred = self._predict_sample(xi)
                target = yi
                
                if y_pred != target:
                    update = self.lr * target
                    self.w[1:] += update * xi
                    self.w[0] += update
                    n_update += 1
               
            

            full_y_pred = np.sign(X.dot(self.w[1:]) + self.w[0])
            print(f'Epoch: {i + 1}, Error: {np.round(np.mean(full_y_pred != y), 3)}')
            print('-----------------------------------\n')
            self.weights.append(self.w)
            self.errors.append(np.mean(full_y_pred != y))
            self.n_updates.append(n_update)

        self.final_w = self.w.copy()
        
        return self

    def _averaged_fit(self, X, y):
        
        self.a = self.w.copy()
        j = 1
        for i in range(self.epochs):
            

            indices = np.arange(self.n_samples)
            np.random.seed(42)
            np.random.shuffle(indices)
            X = X.iloc[indices]
            y = y.iloc[indices]

            for xi, yi in zip(X.values, y.values):
                # print(self.a)
                y_pred = self._vp_predict_sample(xi)
                target = yi

                if y_pred != target:
                    self.weights.append(self.w.copy())

                    self.w[1:] += self.lr * target * xi
                    self.w[0] += self.lr * target
                
                self.a += self.w
                j += 1

            
            full_y_pred = self.predict(X)
            epoch_error = np.mean(full_y_pred != y)
            self.errors.append(epoch_error)
            
            print(f'Epoch: {i+1}, Error: {np.round(epoch_error, 3)}')
            print('-----------------------------------\n')
        
        self.a = self.a / j

        self.final_w = self.a.copy()

        return self
    
    def predict(self, X):
        method = self.method

        if method == 'voted':
            # Voted Prediction

            y_pred = self._vp_predict(X)
            return y_pred
        else:
            if method == 'averaged':
                # Averaged Prediction

                w = self.a
            elif method == 'standard':
                # Standard Prediction

                w = self.w
            y_pred = np.where(X.dot(w[1:]) + w[0] >= 0, 1, -1)
            return y_pred
 
    def _vp_predict(self, X):
        full_y_pred = np.zeros(X.shape[0])
        for w,c in zip(self.weights, self.cs):
            y_pred = np.where(X.dot(w[1:]) + w[0] > 0, 1, -1)
            y_pred = y_pred * c
            full_y_pred += y_pred
            # y_pred = X.dot(w[1:]) + w[0]
            # yield np.where(y_pred > 0, 1, -1)
        
        return np.sign(full_y_pred)
    
    def _vp_predict_sample(self, X):
        y_pred = (X.dot(self.w[1:]) + self.w[0])
        return np.where(y_pred > 0, 1, -1)
    
    def _predict_sample(self, X):
        y_pred = (X.dot(self.w[1:]) + self.w[0])
        return np.where(y_pred >= 0, 1, -1)

    def _initialize_weights(self):
        self.w = np.zeros(self.n_features + 1)
    
    def plot(self):
        plt.plot(range(1, len(self.errors) + 1), self.errors, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Average Prediction Error')
        plt.title(f'{self.method} Perceptron: Training Error vs. Epoch')
        plt.show()
