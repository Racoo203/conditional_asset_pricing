import optuna
from typing import Optional
from sklearn.linear_model import HuberRegressor, SGDRegressor, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.models.config import ModelConfig

class ModelFactory:
    """
    Centralized factory to instantiate models with or without Optuna tuning.
    """
    
    @staticmethod
    def create_model(config: ModelConfig, trial: Optional[optuna.Trial] = None):
        params = config.params.copy()

        if config.use_optuna and trial:
            ModelFactory._inject_optuna_params(config.model_type, params, trial)

        return ModelFactory._build_instance(config, params)

    @staticmethod
    def _inject_optuna_params(model_type: str, params: dict, trial: optuna.Trial):
        """Defines the Search Space for each model."""
        
        if model_type == 'xgboost':
            params['learning_rate'] = trial.suggest_float('learning_rate', 1e-3, 0.3, log=True)
            params['max_depth'] = trial.suggest_int('max_depth', 2, 6) # Reduced max depth to prevent overfitting
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
            params['reg_alpha'] = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True)
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True)
        
        elif model_type == 'random_forest':
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 200)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 12)
            params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 2, 50)
            
        elif model_type == 'pcr':
            params['n_components'] = trial.suggest_int('n_components', 1, 30)
            
        elif model_type == 'sgd_huber':
            params['alpha'] = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
            params['l1_ratio'] = trial.suggest_float('l1_ratio', 0.0, 1.0)
            
        elif model_type == 'huber':
            params['epsilon'] = trial.suggest_float('epsilon', 1.0, 1.9)
            params['alpha'] = trial.suggest_float('alpha', 1e-5, 1e-2, log=True)

        elif model_type == 'mlp':
            params['alpha'] = trial.suggest_float('alpha', 1e-5, 1e-1, log=True)
            params['learning_rate_init'] = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)

    @staticmethod
    def _build_instance(config: ModelConfig, params: dict):
        """Constructs the actual model object with increased iteration limits."""
        
        model_type = config.model_type

        if model_type == 'huber':
            return HuberRegressor(max_iter=2000, **params)
        
        elif model_type == 'sgd_huber':
            return SGDRegressor(loss='huber', penalty='elasticnet', max_iter=2000, **params)
            
        elif model_type == 'pcr':
            return Pipeline([
                ('pca', PCA(n_components=params['n_components'])),
                ('reg', LinearRegression())
            ])
            
        elif model_type == 'random_forest':
            return RandomForestRegressor(n_jobs=-1, **params)
            
        elif model_type == 'xgboost':
            return XGBRegressor(objective='reg:squarederror', n_jobs=-1, **params)
            
        elif model_type == 'mlp':
            layers = tuple([max(4, 2**(6-i)) for i in range(config.n_hidden_layers)])
            
            return MLPRegressor(
                hidden_layer_sizes=layers,
                early_stopping=True,
                max_iter=1000, 
                **params
            )
            
        else:
            raise ValueError(f"Unknown model_type: {model_type}")