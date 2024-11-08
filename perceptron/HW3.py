import numpy as np
import matplotlib.pyplot as plt
from data_process_class import DataProcessor
from perceptron import Perceptron

train_path = './data/bank-note/train.csv'
dp = DataProcessor(train_path, is_banknote = True)
X_train, y_train = dp.X, dp.y

test_path = './data/bank-note/test.csv'
dp2 = DataProcessor(test_path, is_banknote = True)
X_test, y_test = dp2.X, dp2.y

def run_perceptron(method, n_epochs = 10):
    p = Perceptron(n_epochs = n_epochs)
    p.fit(X_train, y_train, method = method)
    if method != 'voted':
        print(f'Final Weights: {np.round(p.final_w, 3)}')
    else:
          [print(f'Weight: {np.round(w, 2)}; c: {c}') for w, c in p.final_w]
    
    print('\n')
    train_pred = p.predict(X_train)
    train_error = np.mean(train_pred != y_train)
    print(f'Training Error: {np.round(train_error,4)}')

    test_pred = p.predict(X_test)
    test_error = np.mean(test_pred != y_test)
    print(f'Test Error: {np.round(test_error,4)}')

    print('\nDo you want to plot the training error? (y/n)')
    choice = input()
    if choice == 'y':
        p.plot()
    
def compare_perceptron_methods(n_epochs = 10):
    print('Comparing Perceptron Methods')
    print('------------------\n')

    print('Standard Perceptron')
    p1 = Perceptron(n_epochs = n_epochs)
    p1.fit(X_train, y_train, method = 'standard')

    print('Voted Perceptron')
    p2 = Perceptron(n_epochs = n_epochs)
    p2.fit(X_train, y_train, method = 'voted')

    print('Averaged Perceptron')
    p3 = Perceptron(n_epochs = n_epochs)
    p3.fit(X_train, y_train, method = 'averaged')

    test_pred1 = p1.predict(X_test)
    test_error1 = np.mean(test_pred1 != y_test)
    test_pred2 = p1.predict(X_test)
    test_error2 = np.mean(test_pred2 != y_test)
    test_pred3 = p1.predict(X_test)
    test_error3 = np.mean(test_pred3 != y_test)

    print(f'Standard Test Error: {np.round(test_error1,4)}')
    print(f'Voted Test Error: {np.round(test_error2,4)}')
    print(f'Averaged Test Error: {np.round(test_error3,4)}')
    
    print('\nDo you want to plot the training errors? (y/n)')
    choice = input()
    if choice == 'y':
       plot_errors(p1, p2, p3, n_epochs = n_epochs)

def plot_errors(p1, p2, p3, n_epochs = 10):
    epochs = range(1, n_epochs + 1)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(epochs, p1.errors, label='Standard Perceptron')
    plt.plot(epochs, p2.errors, label='Voted Perceptron')
    plt.plot(epochs, p3.errors, label='Averaged Perceptron')
    
    plt.xlabel('Epochs')
    plt.ylabel('Training Error')
    plt.title('Training Error vs Epochs for Different Perceptron Methods')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def ask_user():
       print('What would you like to do?')
       print('1. Run Standard Perceptron')
       print('2. Run Voted Perceptron')
       print('3. Run Averaged Perceptron')
       print('4. Compare Perceptron Methods')
       print('5. Exit')
       choice = input()
       return choice

def main():
    while True:
       choice = ask_user()
       if choice == '1':
              print('\nStandard Perceptron')
              print('------------------')
              print('How many trees epochs you like? (def: 10)')
              try:
                     n_epochs = int(input())
              except:
                     n_epochs = 10
              run_perceptron(n_epochs=n_epochs, method='standard')
              print('\n')

       elif choice == '2':
              print('\nVoted Perceptron')
              print('------------------')
              print('How many trees epochs you like? (def: 10)')
              try:
                     n_epochs = int(input())
              except:
                     n_epochs = 10
              run_perceptron(n_epochs=n_epochs, method='voted')
              print('\n')

       elif choice == '3':
              print('\nAveraged Perceptron')
              print('------------------')
              print('How many trees epochs you like? (def: 10)')
              try:
                     n_epochs = int(input())
              except:
                     n_epochs = 10
              run_perceptron(n_epochs=n_epochs, method='averaged')
              print('\n')


       elif choice == '4':
                print('\nComparing Perceptron Methods')
                print('------------------')
                print('How many trees epochs you like? (def: 10)')
                try:
                         n_epochs = int(input())
                except:
                         n_epochs = 10
                compare_perceptron_methods(n_epochs=n_epochs)
                print('\n')
       elif choice == '5':
              break
       else:
            print('Invalid choice. Please try again.')

if __name__ == '__main__':
    main()