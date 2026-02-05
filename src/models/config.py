from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Literal

@dataclass
class ModelConfig:
    name: str
    model_type: Literal['huber', 'sgd_huber', 'pcr', 'random_forest', 'xgboost', 'mlp']
    feature_set: str 
    params: Dict[str, Any] = field(default_factory=dict)
    use_optuna: bool = False
    optuna_trials: int = 20
    description: Optional[str] = None

    @classmethod
    def OLS_Huber(cls, name: str, feature_set: str = 'all', epsilon: float = 1.35):
        """Creates a standard OLS-Huber configuration."""
        return cls(
            name = name,
            model_type = 'huber',
            feature_set = feature_set,
            params = {'epsilon': epsilon},
            use_optuna = False
        )

    @classmethod
    def ElasticNet_Huber(cls, name: str, feature_set: str = 'all', trials: int = 20):
        """Creates a Tunable ElasticNet-Huber configuration."""
        return cls(
            name = name,
            model_type = 'sgd_huber',
            feature_set = feature_set,
            params = {'alpha': 0.0001, 'l1_ratio': 0.15},
            use_optuna = True,
            optuna_trials = trials
        )
    
    @classmethod
    def PCR(cls, name: str, n_components: int = 10, trials: int = 20):
        """Principal Component Regression."""
        return cls(
            name = name, 
            model_type = 'pcr', 
            feature_set = 'all',
            params = {'n_components': n_components},
            use_optuna = True,
            optuna_trials = trials
        )
        
        
    @classmethod
    def RandomForest(cls, name: str, trees: int = 100, depth: int = 3, trials: int = 20):
        """Random Forest."""
        return cls(
            name = name, 
            model_type = 'random_forest', 
            feature_set = 'all',
            params = {'n_estimators': trees, 'max_depth': depth, 'n_jobs': -1},
            use_optuna = True,
            optuna_trials = trials
        )

    @classmethod
    def XGBoost(cls, name: str, trials: int = 20):
        """XGBoost Regressor."""
        return cls(
            name = name, 
            model_type = 'xgboost', 
            feature_set = 'all',
            params = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'n_jobs': -1},
            use_optuna = True,
            optuna_trials = trials
        )

    @classmethod
    def MLP(cls, name: str, n_hidden_layers : int = 3, trials: int = 20):
        """Multilayer Perceptron - 1 Hidden Layer."""
        return cls(
            name = name, 
            model_type = 'mlp', 
            feature_set = 'all',
            params = {'n_hidden_layers': n_hidden_layers},
            use_optuna = True,
            optuna_trials = trials
        )