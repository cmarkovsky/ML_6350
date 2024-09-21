import pandas as pd
from tree_class import Tree

def percent_correct(predict, y):
    # print(predict)
    n_correct = 0
    for i in range(len(predict)):
        if predict.iloc[i] == y.iloc[i]:
            n_correct += 1
    return 1 - (n_correct / len(predict))

### SET UP YOUR RUN

## data = 'car' or 'bank'
data = 'car'

## set = 'train' or 'test
set = 'train'

## metric = 'EN' (info gain), 'GI' (gini index), or 'ME' (majority error)
metric = 'EN'

## min_depth = minimum depth to try
min_depth = 1

## max_depth = maximum depth to try
max_depth = 6

if data == 'car':
    col = ['buying','maint','doors','persons','lug_boot','safety','label']
    features = col[:-1]
    labels = ['unacc', 'acc', 'good', 'vgood']
    df_train = pd.read_csv(f'./data/{data}/{set}.csv', names = col)

    X_train = df_train.drop(columns=['label'])
    y_train = df_train['label']

    df_test = pd.read_csv('data/car/test.csv', names = col)

    X_test = df_test.drop(columns=['label'])
    y_test = df_test['label']

elif data == 'bank':
    col = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign',
       'pdays','previous', 'poutcome', 'y']
    features = col[:-1]
    labels = ['no', 'yes']

    df_train = pd.read_csv('data/bank/train.csv', names = col)

    X_train = df_train.drop(columns=['y'])
    y_train = df_train['y']
    df_test = pd.read_csv('data/bank/test.csv', names = col)

    X_test = df_test.drop(columns=['y'])
    y_test = df_test['y']
else:
    print('Please enter a valid dataset')


for i in range(min_depth, max_depth):
    tree = Tree(max_depth=i, gain_metric=metric)
    tree.train_tree(X_train, y_train, features, labels)
    y_predict = tree.predict2(X_test)
    error = percent_correct(y_predict, y_test)
    print(f'Depth: {i}; Error: {error}')
