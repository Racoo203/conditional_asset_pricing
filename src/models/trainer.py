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

# Use shared logger but ensure Optuna propagates logs
logger = setup_logger("Trainer", log_dir="reports/experiments")

class AssetPricingTrainer:
    def __init__(self, df: pd.DataFrame, target_col='target_ret_excess', date_col='date'):
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col
        
        self.ff3_features = ['mvel1', 'bm', 'mom12m'] 
        self.unique_dates = sorted(self.df[self.date_col].unique())
        
        # Directories for persistence
        self.param_dir = "data/params"
        self.model_dir = "models/trained"
        self.optuna_db = "sqlite:///data/params/optuna.db"

        # os.makedirs(self.param_dir, exist_ok=True)
        # os.makedirs(self.model_dir, exist_ok=True)

    def _get_feature_subset(self, feature_set_name: str) -> List[str]:
        if feature_set_name == 'ff3':
            return self.ff3_features
        elif feature_set_name == 'all':
            exclude = {self.target_col, self.date_col, 'date_fmt', 'permno', 'year'}
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

    def _load_best_params(self, model_name: str) -> Dict[str, Any]:
        """Tries to load params from disk. Returns empty dict if not found."""
        path = os.path.join(self.param_dir, f"{model_name}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                logger.info(f"   > Loaded persisted params from {path}")
                return json.load(f)
        return {}

    def _save_best_params(self, model_name: str, params: Dict[str, Any]):
        """Saves best params to disk."""
        path = os.path.join(self.param_dir, f"{model_name}.json")
        with open(path, 'w') as f:
            json.dump(params, f, indent=4)
        logger.info(f"   > Saved best params to {path}")

    def _tune_hyperparameters(self, df_dev: pd.DataFrame, config: ModelConfig) -> Dict[str, Any]:
        """
        Step 2: Hyperparameter Tuning via Optuna.
        Smart Resume: Checks DB first. If target trials reached, skips tuning.
        """
        # 1. If Tuning is explicitly DISABLED in config, look for JSON file
        if not config.use_optuna:
            saved_params = self._load_best_params(config.name)
            final_params = config.params.copy()
            final_params.update(saved_params)
            return final_params

        # 2. Setup Study (Connect to DB)
        logger.info(f"   > Checking Optuna state for {config.name}...")
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        study = optuna.create_study(
            study_name=f"{config.name}_optimization",
            storage=self.optuna_db,
            direction='maximize',
            load_if_exists=True 
        )
        
        # 3. SMART RESUME LOGIC
        # Count only COMPLETE trials to avoid counting crashed/pruned ones
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        n_existing = len(completed_trials)
        n_remaining = config.optuna_trials - n_existing
        
        if n_remaining > 0:
            logger.info(f"   > Resuming: Found {n_existing} trials. Running {n_remaining} more...")
            
            def objective(trial):
                return self.cross_validate(df_dev, config, n_splits=3, trial=trial)
            
            study.optimize(objective, n_trials=n_remaining)
            logger.info(f"   > Tuning Complete.")
            
        else:
            logger.info(f"   > Target Reached: Found {n_existing} trials (Target: {config.optuna_trials}). SKIPPING tuning.")

        # 4. Save and Return Best Params
        # Even if we skipped tuning, we still fetch the best params from the DB
        try:
            logger.info(f"   > Best Value: {study.best_value:.5f}")
            logger.info(f"   > Best Params: {study.best_params}")
            
            best_params = config.params.copy()
            best_params.update(study.best_params)
            self._save_best_params(config.name, best_params)
            return best_params
            
        except ValueError:
            # Matches case where n_trials=0 and DB is empty
            logger.warning("   ! No completed trials found. Using default params.")
            return config.params

    def _train_final_model(
            self, 
            df_dev: pd.DataFrame, 
            config: ModelConfig, 
            best_params: Dict[str, Any], 
            force_retrain: bool = False
        ):
        """Step 3: Train final model on full development set."""

        model_path = os.path.join(self.model_dir, f"{config.name}.joblib")
        
        # 1. Check if model exists and we are not forcing a retrain
        if os.path.exists(model_path) and not force_retrain:
            logger.info(f"   > Loading pre-trained model from {model_path}...")
            try:
                model = joblib.load(model_path)
                return model
            except Exception as e:
                logger.warning(f"   ! Failed to load model ({e}). Retraining...")

        logger.info(f"   > Training Final Model ({config.name})...")
        features = self._get_feature_subset(config.feature_set)
        
        final_config = ModelConfig(
            name=config.name,
            model_type=config.model_type,
            feature_set=config.feature_set,
            params=best_params,
            use_optuna=False
        )
        
        model = ModelFactory.create_model(final_config)
        model.fit(df_dev[features], df_dev[self.target_col])
        return model

    def _evaluate_model(self, model, df_test: pd.DataFrame, config: ModelConfig):
        """Step 4: Out-of-Sample Evaluation."""
        features = self._get_feature_subset(config.feature_set)
        y_test = df_test[self.target_col]
        y_pred = model.predict(df_test[features])
        
        metrics = regression_metrics(y_test, y_pred)
        
        logger.info(f"RESULT {config.name}: R2_OOS={metrics['r2_oos']:.5f} | RMSE={metrics['rmse']:.5f}")
        return metrics

    def run_experiment(self, config: ModelConfig, force_retrain: bool = False):
        """Orchestrates the full ML Pipeline."""
        logger.info(f"--- STARTING EXPERIMENT: {config.name} ---")
        df_dev, df_test = self.strict_time_split(self.df)
        best_params = self._tune_hyperparameters(df_dev, config)
        model = self._train_final_model(df_dev, config, best_params, force_retrain=force_retrain)
        metrics = self._evaluate_model(model, df_test, config)
        
        return metrics