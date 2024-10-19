import numpy as np

from data_process_class import DataProcessor
from grad_descent_class import GradientDescent



train_path = './data/concrete/train.csv'
dp_train = DataProcessor(train_path, is_bank=False)
X_train, y_train = dp_train.X, dp_train.y

test_path = './data/concrete/test.csv'
dp_test = DataProcessor(test_path, is_bank = False)
X_test, y_test = dp_test.X, dp_test.y

def run_batch_gd(lr = 0.1, n_iter = 100):
       gd = GradientDescent(learning_rate = lr, max_iter = n_iter)
       gd.batch_fit(X_train, y_train)
       print(f'Learning Rate: {lr}')
       print(f'Weights: {np.round(gd.weights, 3)}')
       print(f'Test Cost: {np.round(gd.calc_test_cost(X_test, y_test), 3)}')
       gd.plot_cost()
       

def run_stochastic_gd(lr = 0.01, n_iter = 2500):
       gd = GradientDescent(learning_rate = lr, max_iter=n_iter)
       gd.stochastic_fit(X_train, y_train)
       print(f'Learning Rate: {lr}')
       print(f'Weights: {np.round(gd.weights, 3)}')
       print(f'Test Cost: {np.round(gd.calc_test_cost(X_test, y_test), 3)}')
       gd.plot_cost()
       
       
def ask_user():
       print('What would you like to do?')
       print('1. Run Batch Gradient Descent')
       print('2. Run Stochastic Gradient Descent')
       print('3. Exit')
       choice = input()
       return choice

def main():
    while True:
       choice = ask_user()
       if choice == '1':
              print('How many iterations would you like to run? (def: 100)')
              try:
                     n_iters = int(input())
              except:
                     n_iters = 100
              print('What learning rate would you like to use? (def: 0.1)')
              try:
                     lr = float(input())
              except:
                     lr = 0.1
              run_batch_gd(lr = lr, n_iter = n_iters)
       elif choice == '2':
              print('How many iterations would you like to run? (def: 1000)')
              try:
                     n_iters = int(input())
              except:
                     n_iters = 1000
              print('What learning rate would you like to use? (def: 0.01)')
              try:
                     lr = float(input())
              except:
                     lr = 0.01
              run_stochastic_gd(lr = lr, n_iter = n_iters)
       elif choice == '3':
              break
       else:
            print('Invalid choice. Please try again.')

if __name__ == '__main__':
    main()