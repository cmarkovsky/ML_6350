import numpy as np
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class DualSVM:
    def __init__(self, C=1.0, gamma=0.1):
        self.C = C
        self.gamma = gamma

    def _gaussian_kernel(self, x, gamma=0.1):
        x = np.array(x)
        sq_dists = np.sum(x**2, axis=1).reshape(-1, 1) + np.sum(x**2, axis=1) - 2 * np.dot(x, x.T)
        K = np.exp(-sq_dists / gamma)
        return K
    
    def _gaussian_kernel2(self, x, gamma=.01):
        x = np.array(x)
        K = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                
                print(f'x1: {x[i]}'), print(f'x2: {x[j]}')
                x1 = x[i].ravel()
                x2 = x[j].ravel()
                k = np.exp(-np.sum(np.square(x1 - x2)) / self.gamma)

        return K
    
    def gaussian_kernel(self, x1, x2):
        return np.exp(-np.linalg.norm(x1 - x2)**2 / self.gamma)
    
    def dual_fit(self, X, y, Gaussian = True):

        self.n_samples, self.n_features = X.shape

        if Gaussian:
            K = np.zeros((self.n_samples, self.n_samples))
            for i in range(len(X)):
                for j in range(len(X)):
                    K[i, j] = self.gaussian_kernel(X[i], X[j])
        else:
            K = np.dot(X, X.T)

        def objective(alpha):
            return 0.5 * np.sum((alpha * y)[:, None] * (alpha * y)[None, :] * K) - np.sum(alpha)

        def eq_constraint(alpha):
            return np.dot(alpha, y)
        
        

        bounds = [(0, self.C) for _ in range(self.n_samples)]

        initial_alpha = np.zeros(self.n_samples)
        result = minimize(objective, initial_alpha, bounds=bounds, constraints={'type': 'eq', 'fun': eq_constraint}, method='SLSQP')
        alphas = result.x

        sv = alphas > 1e-5
        self.alphas = alphas[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        self.b = np.mean([y_k - np.sum(self.alphas * self.sv_y * K[sv_k, sv]) for sv_k, y_k in zip(np.where(sv)[0], self.sv_y)])
        self.w = np.sum(self.alphas[:, None] * self.sv_y[:, None] * self.sv, axis=0)

    def dual_predict(self, X, Gaussian = True):
        if not Gaussian:
            return np.sign(np.dot(X, self.w) + self.b)
        else:
            y_predict = np.zeros(len(X))
            for i in range(len(X)):
                kernel = [self.alphas[j] * self.sv_y[j] * self.gaussian_kernel(X[i], self.sv[j]) for j in range(len(self.sv))]
                y_predict[i] = np.sum(kernel)
            return np.sign(y_predict + self.b)  
      
    def calculate_error(self, X, y):
        return np.mean(self.dual_predict(X) != y)
