#src/models/validator.py
import numpy as np
import pandas as pd
from typing import Callable


class BaseValidator:
    def score(self, build_fn: Callable[[dict], object], params: dict, df_dev: pd.DataFrame, feature_set: str, target_col: str, date_col: str, n_splits: int = 5):
        raise NotImplementedError


class RollingWindowValidator(BaseValidator):
    def __init__(self, target_col: str = 'target_ret_excess', date_col: str = 'date', n_splits: int = 5):
        self.target_col = target_col
        self.date_col = date_col
        self.n_splits = n_splits

    def score(self, build_fn: Callable[[dict], object], params: dict, df_dev: pd.DataFrame, feature_set: str, target_col: str = None, date_col: str = None, n_splits: int = None):
        # allow overriding via args
        deltas = []

        if target_col is None:
            target_col = self.target_col
        if date_col is None:
            date_col = self.date_col
        if n_splits is None:
            n_splits = self.n_splits

        features = [c for c in df_dev.columns if c not in {target_col, date_col, 'date_fmt', 'permno', 'year', 'join_month', 'mktcap_next', 'sic2'}]
        dates = sorted(df_dev[date_col].unique())
        fold_size = len(dates) // (n_splits + 1) if (n_splits + 1) > 0 else len(dates)

        scores_r2 = []
        scores_rmse = []
        n_folds = 0

        for i in range(1, n_splits + 1):

            train_end_idx = i * fold_size
            val_end_idx = train_end_idx + fold_size

            train_dates = dates[:train_end_idx]
            val_dates = dates[train_end_idx:val_end_idx]

            train_mask = df_dev[date_col].isin(train_dates)
            val_mask = df_dev[date_col].isin(val_dates)

            if not val_mask.any():
                continue

            X_train = df_dev.loc[train_mask, features]
            y_train = df_dev.loc[train_mask, target_col]
            X_val = df_dev.loc[val_mask, features]
            y_val = df_dev.loc[val_mask, target_col]

            model = build_fn(params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)

            mse = np.mean((y_val - preds) ** 2)
            rmse = np.sqrt(mse)
            # R2 OOS (financial definition)
            mse_zero = np.mean(np.square(y_val)) if len(y_val) else np.nan
            r2_oos = 1 - (mse / mse_zero) if mse_zero != 0 else float('nan')

            scores_r2.append(r2_oos)
            scores_rmse.append(rmse)
            n_folds += 1

        mean_r2 = float(np.nanmean(scores_r2)) if scores_r2 else float('nan')
        mean_rmse = float(np.nanmean(scores_rmse)) if scores_rmse else float('nan')

        return {
            'mean_r2': mean_r2,
            'mean_rmse': mean_rmse,
            'r2s': scores_r2,
            'rmses': scores_rmse,
            'n_folds': n_folds,
        }