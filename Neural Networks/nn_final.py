import numpy as np
from data_process_class import DataProcessor
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, n_features, n_neurons, n_layers, lr0, d, init_zero = False, seed=0):
        self.n_features = n_features
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.init_zero = init_zero
        self.lr0 = lr0
        self.d = d
        self.learning_rate = lr0
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        self.network = self._init_network()
       
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def update_lr(self, epoch):
        self.learning_rate = self.lr0 * (1 / (1 + self.lr0 * epoch / self.d))
    
    def _init_weights(self):
        weights = []
        weights.append(np.random.normal(0, 1, (self.n_neurons, self.n_features + 1)))
        for i in range(1, self.n_layers):
            weights.append(np.random.normal(0, 1, (self.n_neurons, self.n_neurons + 1))) 
        weights.append(np.random.normal(0, 1, (1, self.n_neurons + 1)))
        return weights
    
    def _init_weights0(self):
        weights = []
        weights.append(np.zeros((self.n_neurons, self.n_features + 1)))
        for i in range(1, self.n_layers):
            weights.append(np.zeros((self.n_neurons, self.n_neurons + 1)))
        weights.append(np.zeros((1, self.n_neurons + 1)))
        return weights
    
    def _init_network(self):
        network = []
        if self.init_zero:
            weights = self._init_weights0()
        else:
            weights = self._init_weights()

        for i in range(self.n_layers):
            network.append({'weights': weights[i], 'activation': 'sigmoid', 'n_neurons': self.n_neurons})
        network.append({'weights': weights[-1], 'activation': 'linear', 'n_neurons': 1})

        return network
    
    def forward_pass(self, x):
        inputs = np.append(1, x) 

        for layer in self.network[:-1]:
            new_inputs = []
            for i in range(layer['n_neurons']):
                z = np.dot(layer['weights'][i], inputs)
                activation = self.sigmoid(z)
                new_inputs.append(activation)
            inputs = np.append(1, new_inputs)
        
        z = np.dot(self.network[-1]['weights'], inputs)
        return z
    
    def calc_loss(self, y, y_pred):
        return 0.5 * (y - y_pred) ** 2
    
    def back_prop(self, x, y):
        inputs = [np.append(1, x)]
        for layer in self.network[:-1]:
            new_inputs = []
            for i in range(layer['n_neurons']):
                z = np.dot(layer['weights'][i], inputs[-1])
                activation = self.sigmoid(z)
                new_inputs.append(activation)
            inputs.append(np.append(1, new_inputs))
        
        z = np.dot(self.network[-1]['weights'], inputs[-1])
        y_pred = z
        
        error = y - y_pred
        delta = error
        
        for i in reversed(range(len(self.network))):
            layer = self.network[i]
            if i != len(self.network) - 1:
                delta = delta * self.sigmoid_derivative(np.array(inputs[i + 1][1:]))
            layer_error = np.dot(delta, layer['weights'])
            layer['weights'] += self.learning_rate * np.outer(delta, inputs[i])
            delta = layer_error[:-1] 
        
        return error
    
    def stochastic_gradient_descent(self, X, y, n_epochs=100):
        self.errors = []
        for epoch in range(n_epochs):
            self.update_lr(epoch + 1)
            if (epoch + 1) % 25 == 0:
                print(f"Epoch {epoch + 1}/{n_epochs}")
                print(f"Loss: {self.errors[-1][0]:.3f}")
            indices = np.arange(len(X))
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            for i in range(len(X)):
                error = self.back_prop(X[i], y[i])
            self.errors.append(error)

    def train(self, X, y):
        self.stochastic_gradient_descent(X, y)            
        
    def print_weights(self):
        for layer in self.network:
            print(layer['weights'])

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self.forward_pass(x))
        return np.array(y_pred)
    
    def plot_loss(self):
        plt.plot(self.errors)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
