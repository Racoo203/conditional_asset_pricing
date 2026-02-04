import pandas as pd
import numpy as np
import optuna
from sklearn.linear_model import HuberRegressor, SGDRegressor
from sklearn.preprocessing import StandardScaler
from src.evaluation.metrics import regression_metrics
from src.models.config import ModelConfig
from tqdm import tqdm

class AssetPricingTrainer:
    def __init__(self, df, target_col='target_ret_excess', date_col='date'):
        self.df = df
        self.target_col = target_col
        self.date_col = date_col
        self.ff3_features = ['mvel1', 'bm', 'mom12m'] # Define your "3 factors"
        
    def _get_features(self, feature_set_name):
        """Selector for feature subsets."""
        if feature_set_name == 'ff3':
            return self.ff3_features
        elif feature_set_name == 'all':
            # Exclude metadata and target
            exclude = [self.target_col, self.date_col, 'date_fmt', 'permno']
            return [c for c in self.df.columns if c not in exclude]
        else:
            raise ValueError(f"Unknown feature set: {feature_set_name}")

    def _get_model_instance(self, config: ModelConfig, trial=None):
        """Factory pattern to instantiate models."""
        params = config.params.copy()
        
        # --- OPTUNA HYPERPARAM INJECTION ---
        if config.use_optuna and trial:
            if config.model_type == 'sgd_huber':
                params['alpha'] = trial.suggest_float('alpha', 1e-4, 1e-1, log=True)
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
            elif config.model_type == 'huber':
                params['epsilon'] = trial.suggest_float('epsilon', 1.0, 2.0)

        # --- MODEL INSTANTIATION ---
        if config.model_type == 'huber':
            # HuberRegressor is robust to outliers (Heavy Tails)
            return HuberRegressor(**params)
        
        elif config.model_type == 'sgd_huber':
            # SGD allows ElasticNet + Huber Loss
            return SGDRegressor(loss='huber', penalty='elasticnet', **params)
            
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

    def train_test_split(self, test_size=0.2):
        """Strict Time-Based Split."""
        dates = sorted(self.df[self.date_col].unique())
        split_idx = int(len(dates) * (1 - test_size))
        split_date = dates[split_idx]
        
        train_val_mask = self.df[self.date_col] < split_date
        test_mask = self.df[self.date_col] >= split_date
        
        return self.df[train_val_mask], self.df[test_mask]

    def cross_validate(self, df_dev, config: ModelConfig, n_splits=5):
        """Rolling CV on Development Set."""
        features = self._get_features(config.feature_set)
        dates = sorted(df_dev[self.date_col].unique())
        fold_size = len(dates) // (n_splits + 1)
        
        scores = []
        
        # Rolling Window Loop (Strictly Date-Based)
        for i in range(1, n_splits + 1):
            train_end = i * fold_size
            val_end = train_end + fold_size
            
            train_dates = dates[:train_end]
            val_dates = dates[train_end:val_end]
            
            # Masking
            train_df = df_dev[df_dev[self.date_col].isin(train_dates)]
            val_df = df_dev[df_dev[self.date_col].isin(val_dates)]
            
            if len(val_df) == 0: continue

            # Fit
            model = self._get_model_instance(config)
            model.fit(train_df[features], train_df[self.target_col])
            
            # Predict
            preds = model.predict(val_df[features])
            metrics = regression_metrics(val_df[self.target_col], preds)
            scores.append(metrics['r2_oos'])
            
        return np.mean(scores)

    def run_experiment(self, config: ModelConfig):
        """
        1. Split Dev/Test
        2. Optuna Search (if enabled) on Dev
        3. Retrain Best on Full Dev
        4. Final Eval on Test
        """
        print(f"\nSTARTING EXPERIMENT: {config.name}")
        df_dev, df_test = self.train_test_split()
        features = self._get_features(config.feature_set)
        
        best_params = config.params
        
        # 1. OPTUNA TUNING
        if config.use_optuna:
            print("   ... Tuning Hyperparameters")
            def objective(trial):
                return self.cross_validate(df_dev, config, n_splits=5) # Reduced splits for speed
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=config.optuna_trials)
            best_params.update(study.best_params)
            print(f"   ... Best Params: {best_params}")

        # 2. FINAL RETRAINING (Train on DEV, Predict TEST)
        final_model_config = ModelConfig(
            name=config.name, 
            model_type=config.model_type, 
            feature_set=config.feature_set,
            params=best_params
        )
        
        model = self._get_model_instance(final_model_config)
        model.fit(df_dev[features], df_dev[self.target_col])
        
        test_preds = model.predict(df_test[features])
        metrics = regression_metrics(df_test[self.target_col], test_preds)
        
        print(f"FINAL TEST RESULTS ({config.name}):")
        print(f"R2_OOS: {metrics['r2_oos']:.5f}")
        print(f"RMSE:   {metrics['rmse']:.5f}")
        
        return metrics