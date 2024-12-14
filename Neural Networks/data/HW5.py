from nn_final import NeuralNetwork
from data_process_class import DataProcessor
import numpy as np

train_path = './data/bank-note/train.csv'
dp = DataProcessor(train_path, is_banknote = True)
X_train, y_train = dp.X.values, dp.y.values

test_path = './data/bank-note/test.csv'
dp2 = DataProcessor(test_path, is_banknote = True)
X_test, y_test = dp2.X.values, dp2.y.values

def run_nn(X_train, y_train, n_neurons, n_layers, lr0, d, init_zero = False):
    nn = NeuralNetwork(n_features=len(X_train[0]), n_neurons=n_neurons, n_layers=n_layers, lr0=lr0, d=d, init_zero=init_zero)
    nn.train(X_train, y_train)
    return nn

import matplotlib.pyplot as plt

def compare_n_nodes(X_train, y_train, n_layers, lr0, d, init_zero = True):
    neurons = [5, 10, 25, 50, 100]
    lr0s = [0.1 , .01, 0.1, 0.1, 0.01]
    ds = [10 , 1, .1, 1, 1]
    fig, axs = plt.subplots(1, len(neurons), figsize=(14, 6))
    train_accs = []
    test_accs = []

    for i, n in enumerate(neurons):
        print(f'\nTraining Neural Network with {n} neurons')
        print('--------------------------------------')
        nn = run_nn(X_train, y_train, n, n_layers, lr0s[i], ds[i], init_zero)
        axs[i].plot(nn.errors)
        axs[i].set_title(f'Neurons: {n}')
        axs[i].set_xlabel('Epochs')
        if i == 0:
            axs[i].set_ylabel('Loss')
        else:
            axs[i].set_ylabel('')
            axs[i].set_yticklabels([])
        print(f'Neurons: {n}, Final Loss: {nn.errors[-1][0]:.3f}')
        train_pred = nn.predict(X_train)
        train_acc = np.mean(np.round(train_pred) == y_train)
        train_accs.append(train_acc)

        test_pred = nn.predict(X_test)
        test_acc = np.mean(np.round(test_pred) == y_test)
        test_accs.append(test_acc)
        print('--------------------------------------\n')
    
    for i, n in enumerate(neurons):
        print(f'Neurons: {n}, Train Error: {1 - train_accs[i]:.3f}, Test Error: {1 - test_accs[i]:.3f}')
    
    
    plt.tight_layout()
    plt.show()
    
    return

def ask_user():
       print('How would you like to test the Neural Network?')
       print('1. Initialize weights with Standard Gaussian Distribution')
       print('2. Initialize weights with zero')
       print('3. Exit')
       choice = input()
       return choice

def main():
    while True:
        choice = ask_user()
        if choice == '1':
            compare_n_nodes(X_train, y_train, 2, 0.01, 1, False)
        elif choice == '2':
            compare_n_nodes(X_train, y_train, 2, 0.01, 1, True)
        elif choice == '3':
            break
        else:
            print('Invalid choice. Please try again.')
            
    return

if __name__ == '__main__':
    main()

