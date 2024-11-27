from data_process_class import DataProcessor
from svm import SVM
import numpy as np
import matplotlib.pyplot as plt
from dual_svm import DualSVM

train_path = './data/bank-note/train.csv'
dp = DataProcessor(train_path, is_banknote = True)
X_train, y_train = dp.X, dp.y
# print(X_train)

test_path = './data/bank-note/test.csv'
dp2 = DataProcessor(test_path, is_banknote = True)
X_test, y_test = dp2.X, dp2.y

def train_primal_svm(X_train, y_train, n_epochs = 100, C = 100/873, schedule = 1, seed = 0, a = .3, gamma_0 = .4):
    np.random.seed(seed)
    inds = np.random.randint(0, len(y_train), n_epochs)
    svm = SVM(n_epochs = n_epochs, C=C, schedule=schedule, gamma_0=gamma_0)
    svm.primal_fit(X_train, y_train, a = a, inds=inds)
    return svm
    # train_error = svm.calculate_error(X_train, y_train)
    # print(f'Training Error: {np.round(train_error,4)}')
    # svm.plot_error()

def compare_svms(X_train, y_train, X_test, y_test, n_epochs = 100, schedule = 1, seeded = True):
    print('Comparing SVMs')
    print('------------------\n')
    print('Would you like to plot the training error? (y/n)')
    choice = input()
    if choice == 'y':
        plot = True
    else:
        plot = False

    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey = True)
    Cs = [100 / 873, 500 / 873, 700 / 873]
    titles = [f'C={np.round(Cs[0],3)}', f'C={np.round(Cs[1],3)}', f'C={np.round(Cs[2],3)}']
    train_errors = []
    test_errors = []
    svms = []

    print('Do you want to seed the random number generator? (y/n)')
    choice = input()
    if choice == 'y':
        seeded = True
    else:
        seeded = False
    
    print('What value of a would you like to use? (def. 0.3)')
    try:
        a = float(input())
    except:
        a = .3
    
    print('What value of gamma_0 would you like to use? (def. 0.4)')
    try:
        gamma_0 = float(input())
    except:
        gamma_0 = .4

    for i, C in enumerate(Cs):
        if seeded:
            seed = 10
        else:
            seed = np.random.randint(0, 1000)
        svm = train_primal_svm(X_train, y_train, n_epochs=n_epochs, C=C, schedule = schedule, seed=seed, a=a, gamma_0=gamma_0)
        svms.append(svm)

        train_error = svm.calculate_error(X_train, y_train)
        train_errors.append(train_error)

        test_error = svm.calculate_error(X_test, y_test)
        test_errors.append(test_error)

        if plot:
            axs[i].plot(range(n_epochs), svm.errors, label='Training Error')
            axs[i].set_title(titles[i])
            axs[i].set_xlabel('Epochs')
            axs[0].set_ylabel('Error')
            axs[i].legend()
            fig.tight_layout()
            if schedule == 1:
                schedule_str = f'$\gamma_t = \gamma_0/(1+\gamma_0*t/a)$'
            else:
                schedule_str = f'$\gamma_t = \gamma_0/(1+t)$'
            fig.suptitle(f'Primal SVM: {schedule_str}, $\gamma_0$ = {svm.gamma_0}, $a$ = {svm.a}', fontsize=14)
    print("\nFinal Results:")
    for i in range(3):
        print(f'{titles[i]}; Training Error: {np.round(train_errors[i], 3)}')
        print(f'{titles[i]}; Test Error: {np.round(test_errors[i], 3)}')

    if plot:
        plt.tight_layout()
        plt.show()
    
    return svms

def compare_svm_params(svms):
    print('\n')

    for i, svm in enumerate(svms):
        print(f'C={np.round(svm.C,3)}, gamma_0={svm.gamma_0}, a={svm.a}')
        print(f'Final Weights: {np.round(svm.final_weights.values,3)}')
        print(f'Final Error: {svm.errors[-1]}')
        print('--------------------------------------------')

def ask_user():
       print('\nWhat would you like to do?')
       print('1. Run Primal SVM')
       print('2. Run Dual Perceptron')
       print('3. Exit')
       choice = input()
       return choice

def main():
    while True:
        choice = ask_user()
        if choice == '1':
            print('\nPrimal SVM')
            print('------------------')
            print('Which schedule would you like to use?')
            print('1. Schedule 1')
            print('2. Schedule 2')
            try:
                schedule = int(input())
            except:
                schedule = 1
            
            print(f'Using schedule {schedule}')
            svms = compare_svms(X_train, y_train, X_test, y_test, schedule=schedule)
        
        elif choice == '2':
            print('\nDual SVM')
            print('------------------')
            print('What C value would you like to use? (def. 500/873)')
            print('1. C = 100/873')
            print('2. C = 500/873')
            print('3. C = 700/873')
            try:
                Cin = int(input())
            except:
                Cin = 2

            if Cin == 1:
                C = 100 / 873
            elif Cin == 3:
                C = 500 / 873
            else:
                C = 700 / 873

            
            print('What gamma value would you like to use? (def. 5)')
            print('1. gamma = 0.1')
            print('2. gamma = 0.5')
            print('3. gamma = 1')
            print('4. gamma = 5')
            print('5. gamma = 100')
            try:
                gammain = int(input())
                if not 1 <= gammain <= 5:
                    raise Exception
            except:
                gammain = 4
            
            if gammain == 1:
                gamma = 0.1
            elif gammain == 2:
                gamma = 0.5
            elif gammain == 3:
                gamma = 1
            elif gammain == 5:
                gamma = 100
            else:
                gamma = 5

            print(f'Using C={C}, gamma={gamma}')

            svm = DualSVM(C = C, gamma = gamma)
            svm.dual_fit(X_train.values, y_train.values)
            training_error = svm.calculate_error(X_train.values, y_train.values)
            print(f"Training error: {training_error}")
            test_error = svm.calculate_error(X_test.values, y_test.values)
            print(f"Test error: {test_error}")
            print(f"Support vectors: {len(svm.alphas)}")
            print(f"Final weights: {np.round(svm.w,3)}")
            print(f"Intercept: {svm.b:.3f}")
            print('--------------------------------------------')
        else:
            break

if __name__ == '__main__':
    main()