import numpy as np

from data_process_class import DataProcessor
from bagging_class import BaggedDecisionTrees
from rand_forest_class import RandomForest, compare_feature_set_sizes
from adaboost_class import AdaBoost

train_path = './data/bank/train.csv'
dp_train = DataProcessor(train_path)
X_train, y_train = dp_train.X, dp_train.y

test_path = './data/bank/test.csv'
dp_test = DataProcessor(test_path)
X_test, y_test = dp_test.X, dp_test.y
n_trees = 5

def run_bdt(n_trees, frac = 0.05):
    bdt = BaggedDecisionTrees(n_trees=n_trees)
    bdt.fit(X_train, y_train, frac = frac)
    bdt.plot_errors(X_train, y_train, X_test, y_test)

def bdt_bias_variance(n_trees, frac = 0.05):
       
       bdt = BaggedDecisionTrees(n_trees)
       bdt.fit(X_train, y_train, frac = .1)
       y_pred = bdt.predict(X_test)

       bias = np.mean((y_test - np.mean(y_pred)) ** 2)
       var = np.mean((np.mean(y_pred) - y_pred) ** 2)
       error = bias + var
       print(f'Bias: {bias}')
       print(f'Variance: {var}')
       print(f'Total Error: {error}')

def rf_bias_variance(n_trees, frac = 0.05):
       
       rf = RandomForest(n_features = 2, n_trees = n_trees)
       rf.fit(X_train, y_train, frac = .5)
       y_pred = rf.predict(X_test)
       # print(y_pred)

       bias = np.mean((y_test - np.mean(y_pred)) ** 2)
       var = np.mean((np.mean(y_pred) - y_pred) ** 2)
       error = bias + var
       print(f'Bias: {bias}')
       print(f'Variance: {var}')
       print(f'Total Error: {error}')

def run_rf(n_trees):
    compare_feature_set_sizes(X_train, y_train, X_test, y_test, n_trees=n_trees)

def run_ada(n_trees):
       ada = AdaBoost(n_classifiers=n_trees)
       ada.fit(X_train, y_train)
       ada.plot_full_error(X_test, y_test, X_train, y_train)
       ada.plot_stump_errors(X_test, y_test, X_train, y_train)

def ask_user():
       print('What would you like to do?')
       print('1. Run Bagged Decision Trees')
       print('2. Run Random Forest')
       print('3. Run AdaBoost')
       print('4. Exit')
       choice = input()
       return choice

def main():
    while True:
       choice = ask_user()
       if choice == '1':
              print('How many trees would you like to use? (def: 25)')
              try:
                     n_trees = int(input())
              except:
                     n_trees = 25
              run_bdt(n_trees)
       elif choice == '2':
              print('How many trees would you like to use? (def: 25)')
              try:
                     n_trees = int(input())
              except:
                     n_trees = 25
              run_rf(n_trees)
       elif choice == '3':
              print('How many trees would you like to use? (def: 25)')
              try:
                     n_trees = int(input())
              except:
                     n_trees = 25
              run_ada(n_trees)
       elif choice == '4':
              break
       else:
            print('Invalid choice. Please try again.')

if __name__ == '__main__':
    main()