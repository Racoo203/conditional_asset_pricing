import sys
import pandas as pd
import numpy as np
import sqlite3
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm

class RollingBenchmark:
    def __init__(self, db_path, initial_train_window=60):
        self.conn = sqlite3.connect(db_path)
        self.initial_train_window = initial_train_window # Months
        self.models = {
            'OLS': LinearRegression(),
            'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5)
        }
        
    def load_data(self):
        """
        Loads the Gold Panel. 
        Note: ideally we stream this, but for baselines fitting in RAM is usually fine 
        if we drop unnecessary columns.
        """
        print("Loading Gold Panel...")
        query = "SELECT date_fmt, permno, target_ret_excess, " \
                "mvel1, bm, mom12m, mom1m, retvol " \
                "FROM gold_panel ORDER BY date_fmt" 
                # Note: Expand this list to ALL 90+ characteristics for real run
        
        self.df = pd.read_sql(query, self.conn)
        print(F"[INFO] DATA LOADED - MEMORY ALLOCATED: {sys.getsizeof(self.df)}")

        self.df['date'] = pd.to_datetime(self.df['date_fmt'])
        print("[INFO] DATE PARSED")
        
    def run(self):
        # Make splits of the data
        results = []
        tscv = TimeSeriesSplit(n_splits=5)

        test_date = self.df['date'].quantile(0.8)
        test_mask = self.df['date'] >= test_date

        X = self.df.drop(columns=['date', 'date_fmt', 'permno', 'target_ret_excess'])
        y = self.df['target_ret_excess']

        # print("[INFO] FEATURE AND TARGET SETS CREATED")

        # X_test = X[test_mask]
        # y_test = y[test_mask]

        print("[INFO] TEST SETS CREATED")

        X_model = X.loc[~test_mask]
        y_model = y.loc[~test_mask]

        print("[INFO] TRAIN AND VAL SETS INITIALIZED")

        # Train loop
        for fold_idx, (train_idx, val_idx) in enumerate(tqdm(tscv.split(X_model))):

            X_train = X_model.iloc[train_idx]
            y_train = y_model.iloc[train_idx]

            X_val = X_model.iloc[val_idx]
            y_val = y_model.iloc[val_idx]

            for name, model in self.models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            
                epoch_res = {
                    'ts_fold': fold_idx,
                    'model': name,
                    'R2_val': r2_score(y_val, y_pred),
                    'RMSE_val': np.sqrt(mean_squared_error(y_val, y_pred))
                }
                results.append(epoch_res)

        return pd.DataFrame(results)