from data_process_class import DataProcessor
import numpy as np
import matplotlib.pyplot as plt

class SVM:

    def __init__(self, n_epochs = 100, gamma_0 = .3, C = 100/873, schedule = 1):
        self.gamma_0 = gamma_0
        self.epochs = n_epochs
        self.C = C
        self.schedule = schedule
    
    def update_gamma(self, t):
        if self.schedule == 1:
            return self.gamma_0 / (1 + self.gamma_0 * t / self.a)
        else:
            return self.gamma_0 / (1 + t)

    def predict(self, X):
        return (np.dot(X, self.w))
    
    def primal_fit(self, X, y, a = .7, inds = None):

        self.n_samples, self.n_features = X.shape
        self._initialize_model(a)
        # X.insert(0, 'ones', 1)
        self.weights = []
        self.errors = []
        
        print(f'Running SVM with C={np.round(self.C, 3)}, gamma_0={self.gamma_0}, a={self.a}')
        print('--------------------------------------------')
        for epoch in range(self.epochs):
            self.gamma = self.update_gamma(epoch+1)
            if inds is not None:
                indx = inds[epoch]
            else:
                indx = np.random.randint(self.n_samples)
            xi = X.iloc[indx]
            yi = y.iloc[indx]

            condition = yi * np.dot(self.w,xi)
            
            if condition <= 1:
                self.w = (1-self.gamma) * self.w + self.gamma * self.C * self.n_samples * yi * xi
                subgrad = self.w-self.C*self.n_samples*xi*yi
            else:
                self.w = (1-self.gamma) * self.w
                subgrad = self.w
            
            self.weights.append(self.w.copy())

            epoch_error = self.calculate_error(X, y)
            self.errors.append(epoch_error)

        self.final_weights = self.w
        print('--------------------------------------------')
        return
                

    def _initialize_model(self, a = .7):
        self.w = np.zeros(self.n_features)
        self.gamma = self.gamma_0
        self.a = a
    
    def plot_error(self):
        plt.plot(range(1, len(self.errors) + 1), self.errors, marker='o')
        plt.xlabel('Epochs')
        plt.ylabel('Average Prediction Error')
        plt.title(f'SVM: Training Error vs. Epoch')
        plt.show()

    def calculate_error(self, X, y):
        pred =  y * np.dot(self.w, X.T)
        
        full_pred = np.where(pred > 0, 1, 0)
        error = 1 - np.mean(full_pred)
        return error
