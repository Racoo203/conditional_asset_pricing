import numpy as np
import pandas as pd
import optuna
from typing import List, Optional

# Models
from sklearn.linear_model import HuberRegressor, SGDRegressor, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Utils
from src.evaluation.metrics import regression_metrics
from src.models.config import ModelConfig
from src.utils.logger import setup_logger

# Setup Logger for this module
logger = setup_logger("Trainer", log_dir="reports/experiments")

class ModelFactory:
    @staticmethod
    def create_model(config: ModelConfig, trial: Optional[optuna.Trial] = None):
        params = config.params.copy()

        # --- HYPERPARAMETER TUNING BLOCK ---
        if config.use_optuna and trial:
            if config.model_type == 'xgboost':
                params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
                params['max_depth'] = trial.suggest_int('max_depth', 1, 6)
                params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
            
            elif config.model_type == 'pcr':
                params['n_components'] = trial.suggest_int('n_components', 1, 30)
                
            elif config.model_type == 'sgd_huber':
                params['alpha'] = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
                params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
                
            elif config.model_type == 'mlp':
                # Tune network depth/width implicitly or explicitly? 
                # For now, let's stick to the config-defined architecture 
                # but maybe tune alpha (regularization)
                params['alpha'] = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)

        # --- MODEL INSTANTIATION BLOCK ---
        if config.model_type == 'huber':
            return HuberRegressor(**params)
        
        elif config.model_type == 'sgd_huber':
            return SGDRegressor(loss='huber', penalty='elasticnet', **params)
            
        elif config.model_type == 'pcr':
            return Pipeline([
                ('pca', PCA(n_components=params['n_components'])),
                ('reg', LinearRegression())
            ])
            
        elif config.model_type == 'random_forest':
            # Force n_jobs to -1 to parallelize
            return RandomForestRegressor(n_jobs=-1, **params)
            
        elif config.model_type == 'xgboost':
            return XGBRegressor(objective='reg:squarederror', n_jobs=-1, **params)
            
        elif config.model_type == 'mlp':
            # Dynamic layer construction: n=3 -> (8, 4, 2) etc.
            # Defaulting to a safe structure if not provided
            n_layers = params.get('n_hidden_layers', 3) 
            hidden_layers_sizes = tuple([2**(6-n) for n in range(n_layers)])

            # Clean params to remove custom keys before passing to sklearn
            sklearn_params = {k: v for k, v in params.items() if k != 'n_hidden_layers'}

            return MLPRegressor(
                hidden_layer_sizes=hidden_layers_sizes, 
                early_stopping=True, 
                **sklearn_params
            )
            
        else:
            raise ValueError(f"Unknown model_type: {config.model_type}")

class AssetPricingTrainer:
    def __init__(self, df: pd.DataFrame, target_col='target_ret_excess', date_col='date'):
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col
        
        self.ff3_features = ['mvel1', 'bm', 'mom12m'] 
        self.unique_dates = sorted(self.df[self.date_col].unique())

    def _get_feature_subset(self, feature_set_name: str) -> List[str]:
        """
        Selects features. 
        """
        if feature_set_name == 'ff3':
            return self.ff3_features
        elif feature_set_name == 'all':
            # Exclude metadata, targets, and the partition column 'year'
            exclude = {self.target_col, self.date_col, 'date_fmt', 'permno', 'year'}
            return [c for c in self.df.columns if c not in exclude]
        else:
            raise ValueError(f"Unknown feature set: {feature_set_name}")

    def strict_time_split(self, df: pd.DataFrame, test_ratio: float = 0.2):
        """Splits data strictly by time to prevent look-ahead bias."""
        split_idx = int(len(self.unique_dates) * (1 - test_ratio))
        split_date = self.unique_dates[split_idx]
        
        train_val_mask = df[self.date_col] < split_date
        test_mask = df[self.date_col] >= split_date
        
        logger.info(f"Split Date: {split_date} | Train Rows: {train_val_mask.sum()} | Test Rows: {test_mask.sum()}")
        return df[train_val_mask], df[test_mask]

    def cross_validate(self, df_dev: pd.DataFrame, config: ModelConfig, n_splits: int = 5, trial=None) -> float:
        """
        Performs Rolling Cross-Validation.
        """
        features = self._get_feature_subset(config.feature_set)
        dates = sorted(df_dev[self.date_col].unique())
        fold_size = len(dates) // (n_splits + 1)
        
        scores = []
        
        for i in range(1, n_splits + 1):
            train_end_idx = i * fold_size
            val_end_idx = train_end_idx + fold_size
            
            train_dates = dates[:train_end_idx]
            val_dates = dates[train_end_idx:val_end_idx]
            
            train_mask = df_dev[self.date_col].isin(train_dates)
            val_mask = df_dev[self.date_col].isin(val_dates)
            
            if not val_mask.any(): continue

            X_train = df_dev.loc[train_mask, features]
            y_train = df_dev.loc[train_mask, self.target_col]
            X_val = df_dev.loc[val_mask, features]
            y_val = df_dev.loc[val_mask, self.target_col]

            model = ModelFactory.create_model(config, trial=trial)
            model.fit(X_train, y_train)
            
            preds = model.predict(X_val)
            metrics = regression_metrics(y_val, preds)
            
            score = metrics['r2_oos']
            scores.append(score)
            
        return np.mean(scores)

    def run_experiment(self, config: ModelConfig):
        logger.info(f"STARTING EXPERIMENT: {config.name}")
        
        df_dev, df_test = self.strict_time_split(self.df)
        features = self._get_feature_subset(config.feature_set)
        
        best_params = config.params.copy()
        
        if config.use_optuna:
            logger.info(f"   ... Tuning with Optuna ({config.optuna_trials} trials)")
            optuna.logging.set_verbosity(optuna.logging.WARNING)
            
            def objective(trial):
                return self.cross_validate(df_dev, config, n_splits=3, trial=trial)
            
            study = optuna.create_study(direction='maximize')
            study.optimize(objective, n_trials=config.optuna_trials)
            
            best_params.update(study.best_params)
            logger.info(f"   ... Best Params: {best_params}")

        # Final Retraining
        final_config = ModelConfig(
            name=config.name,
            model_type=config.model_type,
            feature_set=config.feature_set,
            params=best_params,
            use_optuna=False
        )
        
        model = ModelFactory.create_model(final_config)
        
        logger.info("   ... Fitting Final Model")
        model.fit(df_dev[features], df_dev[self.target_col])
        
        y_test = df_test[self.target_col]
        y_test_pred = model.predict(df_test[features])
        metrics = regression_metrics(y_test, y_test_pred)
        
        logger.info(f"RESULT {config.name}: R2_OOS={metrics['r2_oos']:.5f}, RMSE={metrics['rmse']:.5f}")
        return metrics