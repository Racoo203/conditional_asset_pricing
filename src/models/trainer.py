import numpy as np
import pandas as pd
import optuna
import os
import json
import joblib
from typing import List, Dict, Any

from src.evaluation.metrics import regression_metrics
from src.models.config import ModelConfig
from src.models.factory import ModelFactory
from src.utils.logger import setup_logger
from datetime import datetime

# Use shared logger but ensure Optuna propagates logs
logger = setup_logger("Trainer", log_dir="reports/experiments")

class AssetPricingTrainer:
    def __init__(self, df: pd.DataFrame, target_col='target_ret_excess', date_col='date'):
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col
        
        self.ff3_features = ['mvel1', 'bm', 'mom12m']
        self.ff5_features = []
        self.unique_dates = sorted(self.df[self.date_col].unique())
        
        # Directories for persistence
        self.model_dir = "src/models/trained"
        self.optuna_db = "sqlite:///data/tuning/optuna.db"

    def _get_feature_subset(self, feature_set_name: str) -> List[str]:
        if feature_set_name == 'ff3':
            return self.ff3_features
        elif feature_set_name == 'all':
            exclude = {
                self.target_col, 
                self.date_col, 
                'date_fmt', 
                'permno', 
                'year', 
                'join_month',
                'mktcap_next', # This would give future info to the model.
                'sic2'
            }
            return [c for c in self.df.columns if c not in exclude]
        else:
            raise ValueError(f"Unknown feature set: {feature_set_name}")

    def strict_time_split(self, df: pd.DataFrame, test_ratio: float = 0.2):
        """Splits data strictly by time."""
        split_idx = int(len(self.unique_dates) * (1 - test_ratio))
        split_date = self.unique_dates[split_idx]
        
        train_mask = df[self.date_col] < split_date
        test_mask = df[self.date_col] >= split_date
        
        logger.info(f"Time Split: {split_date.date()} | Train: {train_mask.sum()} | Test: {test_mask.sum()}")
        return df.loc[train_mask], df.loc[test_mask]

    def cross_validate(self, df_dev: pd.DataFrame, config: ModelConfig, n_splits: int = 5, trial=None) -> float:
        """Rolling Cross-Validation."""
        features = self._get_feature_subset(config.feature_set)
        dates = sorted(df_dev[self.date_col].unique())
        fold_size = len(dates) // (n_splits + 1)
        
        scores_r2 = []
        scores_rmse = []
        
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
            
            # Robust extraction
            scores_r2.append(metrics['r2_oos'])
            scores_rmse.append(metrics['rmse'])

        # Report extra metrics to Optuna Dashboard
        if trial:
            trial.set_user_attr("avg_rmse", np.mean(scores_rmse))
            
        return np.mean(scores_r2)

    def _tune_hyperparameters(self, df_dev: pd.DataFrame, config: ModelConfig) -> None:
        """
        Step 2: Hyperparameter Tuning.
        """

        logger.info(f"   > Checking Optuna state for {config.name}...")
        optuna.logging.set_verbosity(optuna.logging.INFO)

        # Smart Resume Logic, Count COMPLETED trials only
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        n_existing = len(completed_trials)
        n_remaining = config.optuna_trials - n_existing
        
        if n_remaining > 0:
            logger.info(f"   > Resuming: Found {n_existing} trials. Running {n_remaining} more...")
            
            def objective(trial):
                return self.cross_validate(df_dev, config, n_splits=1, trial=trial)
            
            self.study.optimize(objective, n_trials=n_remaining)
            logger.info(f"   > Tuning Complete.")
        else:
            logger.info(f"   > Target Reached: Found {n_existing} trials (Target: {config.optuna_trials}). SKIPPING tuning.")

    def _train_final_model(
            self, 
            df_dev: pd.DataFrame, 
            config: ModelConfig, 
        ):
        """
        Step 3: Train final model from best parameters in DB.
        """
        
        logger.info(f"   > Training Final Model ({config.name})...")
        features = self._get_feature_subset(config.feature_set)
        best_params = self.study.best_params
        
        final_config = ModelConfig(
            name=config.name,
            model_type=config.model_type,
            n_hidden_layers=config.n_hidden_layers,
            feature_set=config.feature_set,
            params=best_params,
            use_optuna=False
        )
        
        model = ModelFactory.create_model(final_config)
        model.fit(df_dev[features], df_dev[self.target_col])

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.model_dir, f"{config.name}_{timestamp}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"   > Saved trained model to {model_path}")

        return model

    def _evaluate_model(self, model, df_test: pd.DataFrame, config: ModelConfig):
        """Step 4: Out-of-Sample Evaluation."""
        features = self._get_feature_subset(config.feature_set)
        y_test = df_test[self.target_col]
        y_pred = model.predict(df_test[features])
        
        metrics = regression_metrics(y_test, y_pred)
        
        logger.info(f"RESULT {config.name}: R2_OOS={metrics['r2_oos']:.5f} | RMSE={metrics['rmse']:.5f}")
        return metrics

    def run_experiment(self, config: ModelConfig):
        """Orchestrates the full ML Pipeline."""
        logger.info(f"--- STARTING EXPERIMENT: {config.name} ---")

        self.study = optuna.create_study(
            study_name=f"{config.name}_optimization",
            storage=self.optuna_db,
            direction='maximize',
            load_if_exists=True 
        )

        logger.info(f"[+] SPLITTING DATA")
        df_dev, df_test = self.strict_time_split(self.df)

        logger.info(f"[+] FINDING OPTIMAL HYPERPARAMETERS")
        self._tune_hyperparameters(df_dev, config)

        logger.info(f"[+] LOADING MODEL")
        model = self._train_final_model(df_dev, config)
        metrics = self._evaluate_model(model, df_test, config)
        
        return metrics