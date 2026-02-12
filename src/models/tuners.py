#src/models/tuners.py
import optuna
from typing import Optional, Tuple
from sklearn.model_selection import ParameterGrid

from src.models.spec import ModelSpec
from src.models.search_spaces import (
    HuberSearchSpace, SGDHuberSearchSpace, PCRSearchSpace,
    RandomForestSearchSpace, XGBoostSearchSpace, MLPSearchSpace
)

class BaseTuner:
    """Tuner abstraction. Implementations should return (best_params, metadata).
    metadata may include the study object for Optuna.
    """

    def optimize(
        self,
        spec: ModelSpec,
        build_fn,
        validator,
        df_dev,
        search_space=None,
        n_trials: int = 30,
        study_name: Optional[str] = None,
        storage: Optional[str] = None,
    ) -> Tuple[dict, dict]:
        raise NotImplementedError


class NoTuner(BaseTuner):
    def optimize(self, spec, build_fn, validator, df_dev, search_space=None, n_trials=0, **kwargs):
        # Return base params immediately
        return spec.base_params.copy(), {'method': 'none'}


class GridSearchTuner(BaseTuner):
    def __init__(self, param_grid: dict):
        self.param_grid = list(ParameterGrid(param_grid))

    def optimize(self, spec, build_fn, validator, df_dev, search_space=None, n_trials=None, **kwargs):
        best_score = -float('inf')
        best_params = spec.base_params.copy()

        for p in self.param_grid:
            params = spec.base_params.copy()
            params.update(p)
            res = validator.score(lambda prm: build_fn.create(spec, prm), params, df_dev, spec.feature_set, validator.target_col, validator.date_col)
            if res['mean_r2'] > best_score:
                best_score = res['mean_r2']
                best_params = params

        return best_params, {'method': 'grid'}


class OptunaTuner(BaseTuner):
    def __init__(self, direction: str = 'maximize', storage: Optional[str] = None):
        self.direction = direction
        self.storage = storage

    def _get_search_space_instance(self, model_type: str):
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

    def optimize(self, spec: ModelSpec, build_fn, validator, df_dev, search_space=None, n_trials: int = 30, study_name: Optional[str] = None, storage: Optional[str] = None):
        # If no search_space provided (or search_space is None), short-circuit
        if search_space is None:
            return spec.base_params.copy(), {'method': 'none'}

        study = optuna.create_study(
            study_name=study_name,
            storage=storage or self.storage,
            direction=self.direction,
            load_if_exists=True,
        )

        def objective(trial: optuna.Trial):
            suggested = search_space.suggest(trial)
            params = spec.base_params.copy()
            params.update(suggested)

            # validator returns dict with mean_r2 and mean_rmse
            res = validator.score(lambda prm: build_fn.create(spec, prm), params, df_dev, spec.feature_set, validator.target_col, validator.date_col)

            # Attach additional info to the trial for dashboard
            trial.set_user_attr('mean_rmse', float(res['mean_rmse']))
            trial.set_user_attr('folds_used', int(res.get('n_folds', 0)))

            return float(res['mean_r2'])

        # resume smartly by only running missing trials
        existing_completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        n_existing = len(existing_completed)
        n_remaining = max(0, n_trials - n_existing)

        if n_remaining > 0:
            study.optimize(objective, n_trials=n_remaining)

        best_params = study.best_params if study.best_params else spec.base_params.copy()
        # merge with base params to ensure defaults persist
        merged = spec.base_params.copy()
        merged.update(best_params)

        return merged, {'method': 'optuna', 'study': study}