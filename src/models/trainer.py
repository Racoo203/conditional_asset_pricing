#src/models/trainer.py
import numpy as np
import pandas as pd
import os
import joblib
from typing import List, Optional

from datetime import datetime

from src.models.spec import ModelSpec
from src.models.factory import ModelFactory
from src.models.search_spaces import (
    HuberSearchSpace, SGDHuberSearchSpace, PCRSearchSpace,
    RandomForestSearchSpace, XGBoostSearchSpace, MLPSearchSpace
)
from src.models.tuners import OptunaTuner
from src.models.validator import RollingWindowValidator
from src.utils.logger import setup_logger
from src.evaluation.metrics import regression_metrics

logger = setup_logger("Trainer", log_dir="reports/experiments")


class AssetPricingTrainer:
    def __init__(self, df: pd.DataFrame, target_col: str = 'target_ret_excess', date_col: str = 'date'):
        self.df = df.copy()
        self.target_col = target_col
        self.date_col = date_col

        self.ff3_features = ['mvel1', 'bm', 'mom12m']
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
                'mktcap_next',
                'sic2'
            }
            return [c for c in self.df.columns if c not in exclude]
        else:
            raise ValueError(f"Unknown feature set: {feature_set_name}")

    def strict_time_split(self, df: pd.DataFrame, test_ratio: float = 0.2):
        split_idx = int(len(self.unique_dates) * (1 - test_ratio))
        split_date = self.unique_dates[split_idx]

        train_mask = df[self.date_col] < split_date
        test_mask = df[self.date_col] >= split_date

        logger.info(f"Time Split: {split_date.date()} | Train: {train_mask.sum()} | Test: {test_mask.sum()}")
        return df.loc[train_mask], df.loc[test_mask]

    def _select_search_space(self, model_type: str):
        mapping = {
            'huber': HuberSearchSpace,
            'sgd_huber': SGDHuberSearchSpace,
            'pcr': PCRSearchSpace,
            'random_forest': RandomForestSearchSpace,
            'xgboost': XGBoostSearchSpace,
            'mlp': MLPSearchSpace,
        }
        cls = mapping.get(model_type)
        return cls() if cls else None

    def _train_final_model(self, df_dev: pd.DataFrame, spec: ModelSpec, final_params: dict):
        logger.info(f"   > Training Final Model ({spec.name})...")
        features = self._get_feature_subset(spec.feature_set)

        model = ModelFactory.create(spec, final_params)
        model.fit(df_dev[features], df_dev[self.target_col])

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.model_dir, exist_ok=True)
        model_path = os.path.join(self.model_dir, f"{spec.name}_{timestamp}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"   > Saved trained model to {model_path}")

        return model

    def _evaluate_model(self, model, df_test: pd.DataFrame, spec: ModelSpec):
        features = self._get_feature_subset(spec.feature_set)
        y_test = df_test[self.target_col]
        y_pred = model.predict(df_test[features])

        metrics = regression_metrics(y_test, y_pred)
        logger.info(f"RESULT {spec.name}: R2_OOS={metrics['r2_oos']:.5f} | RMSE={metrics['rmse']:.5f}")
        return metrics

    def run_experiment(self, spec: ModelSpec, n_trials: int = 30, tuner: Optional[object] = None, validator: Optional[object] = None):
        """Orchestrates the ML pipeline using Tuner + Validator abstractions.

        Defaults: Optuna tuner + RollingWindowValidator (keeps previous behaviour).
        """
        logger.info(f"--- STARTING EXPERIMENT: {spec.name} ---")

        df_dev, df_test = self.strict_time_split(self.df)
        print(f"df_dev rows: {len(df_dev)}")
        print(f"Number of unique dates: {len(self.unique_dates)}")

        # choose search space
        search_space = self._select_search_space(spec.model_type)

        # default tuner & validator
        tuner = tuner or OptunaTuner(storage=self.optuna_db)
        validator = validator or RollingWindowValidator(target_col=self.target_col, date_col=self.date_col)

        logger.info(f"[+] FINDING OPTIMAL HYPERPARAMETERS (n_trials={n_trials})")
        best_params, meta = tuner.optimize(
            spec=spec,
            build_fn=ModelFactory,
            validator=validator,
            df_dev=df_dev,
            search_space=search_space,
            n_trials=n_trials,
            study_name=f"{spec.name}_optimization",
            storage=self.optuna_db,
        )

        logger.info(f"[+] LOADING MODEL")
        model = self._train_final_model(df_dev, spec, best_params)

        metrics = self._evaluate_model(model, df_test, spec)
        return metrics