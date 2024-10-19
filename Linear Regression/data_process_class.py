import pandas as pd
import numpy as np

class DataProcessor:
    def __init__(self, path, is_bank = True):
        self.median_dict = {}
        self.features = None
        self.labels = None
        self.path = path
        if is_bank:
            self.X, self.y = self._process_bank_data(self.path)
        else:
            self.X, self.y = self._process_concrete_data(self.path)

    def _process_concrete_data(self, path):
        col = ['cement', 'slag', 'fly_ash', 'water', 'sp', 'coarse_agg', 'fine_agg', 'y']
        self.features = col[:-1]
        df = pd.read_csv(path, names  = col)
        X = df.drop(columns=['y'])

        y = df['y']
        
        return X, y
    def _process_bank_data(self, path):
        col = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign',
            'pdays','previous', 'poutcome', 'y']
        self.features = col[:-1]
        self.labels = ['no', 'yes']
        df = pd.read_csv(path, names  = col)
        X = df.drop(columns=['y'])
        X = self._conv_numerics(X)

        y = df['y']
        y = self._convert_labels(y)
        
        return X, y

    def _conv_numerics(self, X):
            df = pd.DataFrame()
            for feature in self.features:
                numeric = type(X[feature].iloc[0]) != str
                if numeric:
                    df[feature] = (np.median(X[feature]) < X[feature])
                    self.median_dict[feature] = np.median(X[feature])
                else:
                    df[feature] = X[feature]
            return df

    def _convert_labels(self, y):
        return pd.Series(np.where(y == self.labels[1], 1, -1))
