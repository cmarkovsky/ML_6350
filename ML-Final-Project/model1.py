import pandas as pd
import xgboost as xgb
import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import logging
import time

from sklearn.metrics import mean_squared_error, r2_score

logging.basicConfig(filename='model1.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Started")
logging.getLogger().setLevel(logging.WARNING)

class DataPrepper:

    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.load_data()
        self.X_train, self.y_train, self.X_test, self.y_test = self.preprocess_data()
        self.X = pd.concat([self.X_train, self.X_test])
        self.y = pd.concat([self.y_train, self.y_test])

    def load_data(self):
        data =  pd.read_csv(self.data_path).drop(columns = ['Unnamed: 0'])
        data = data.rename(columns = {'dh1': 'target', 'elevation': 'z'})
        return data
    
    def preprocess_data(self):
        x_train = self.data[self.data['void_mask'] == False]
        y_train = x_train['target']
        x_train = x_train.drop(columns = ['RGIId_Full', 'target', 'void_mask'])

        x_test = self.data[self.data['void_mask'] == True]
        y_test = x_test['target']
        x_test = x_test.drop(columns = ['RGIId_Full', 'target', 'void_mask'])
        return x_train, y_train, x_test, y_test
    

# Choose hyperparameter domain to search over
space = {
        'max_depth':hp.choice('max_depth', np.arange(1, 30, 1, dtype=int)),
        'colsample_bytree':hp.quniform('colsample_bytree', 0.3, 1.01, 0.05),
        'min_child_weight':hp.choice('min_child_weight', np.arange(1, 30, 1, dtype=int)),
        'subsample':        hp.quniform('subsample', 0.3, 1.01, 0.05),
        'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 1.01, 0.05)),
        'gamma': hp.quniform('gamma', 0.1, 5, 0.05),
    
        'objective':'reg:squarederror',
        'eval_metric': 'rmse',
    }


dp = DataPrepper("./data/ts1.csv")
X_train, y_train = dp.X_train, dp.y_train
X_test, y_test = dp.X_test, dp.y_test



def score(params, n_folds=5):
    
    #Cross-validation
    d_train = xgb.DMatrix(X_train,y_train)
    
    cv_results = xgb.cv(params, d_train, nfold = n_folds, num_boost_round=500,
                        early_stopping_rounds = 10, metrics = 'rmse', seed = 0)
    
    loss = min(cv_results['test-rmse-mean'])
    
    return loss


def optimize(trials, space):
    
    best = fmin(score, space, algo=tpe.suggest, max_evals=1000, verbose = True, trials=trials)#Add seed to fmin function
    min_loss = min([trial['result']['loss'] for trial in trials.trials])
    return best, min_loss

# Measure the start time
start_time = time.time()

trials = Trials()
best_params, min_loss = optimize(trials, space)

# Measure the end time
end_time = time.time()

# Calculate the duration
duration = end_time - start_time

logging.getLogger().setLevel(logging.INFO)

# Log the final loss and time taken
logging.info(f"Final loss: {min_loss}")
logging.info(f"Time taken: {duration} seconds")

# Return the best parameters
best_params = space_eval(space, best_params)
print("BEST PARAMETERS: " + str(best_params))

data_test = xgb.DMatrix(X_test,y_test)

data = xgb.DMatrix(X_train,y_train)
final_model = xgb.train(best_params, data, num_boost_round=500, verbose_eval=False,
                            evals=[(data_test, "Test")],early_stopping_rounds=10)


data_all = xgb.DMatrix(dp.X)
y_pred = final_model.predict(data_all)


rmse_str = str(np.sqrt(mean_squared_error(dp.y, y_pred)))
logging.info("RMSE: " + rmse_str)

r2_str = str(r2_score(dp.y, y_pred))
logging.info("R2: " + r2_str)

print("RMSE: " + rmse_str)
print("R2: " + r2_str)

logging.info("Finished")

dp.data['dh_pred'] = y_pred
dp.data.to_csv("./results/ts1_pred.csv", index = False)

# print(X_train.head(), y_train.head())