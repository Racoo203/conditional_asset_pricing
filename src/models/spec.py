from dataclasses import dataclass
from typing import Dict, Any, Optional, Literal

ModelType = Literal[
    'huber',
    'sgd_huber',
    'pcr',
    'random_forest',
    'xgboost',
    'mlp'
]

@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_type: ModelType
    feature_set: str
    base_params: Dict[str, Any]
    architecture: Optional[Dict[str, Any]] = None
    description: Optional[str] = None

    # --- Named constructors (still OK, but now clean) ---

    @classmethod
    def OLS_Huber(cls, name: str, feature_set: str = 'all'):
        return cls(
            name=name,
            model_type='huber',
            feature_set=feature_set,
            base_params={'epsilon': 1.35, 'alpha': 1e-4},
            description="OLS with Huber loss"
        )

    @classmethod
    def ElasticNet_Huber(cls, name: str, feature_set: str = 'all'):
        return cls(
            name=name,
            model_type='sgd_huber',
            feature_set=feature_set,
            base_params={'alpha': 1e-4, 'l1_ratio': 0.15},
            description="ElasticNet with Huber loss"
        )

    @classmethod
    def PCR(cls, name: str, feature_set: str = 'all'):
        return cls(
            name=name,
            model_type='pcr',
            feature_set=feature_set,
            base_params={'n_components': 3},
            description="Principal Component Regression"
        )

    @classmethod
    def RandomForest(cls, name: str, feature_set: str = 'all'):
        return cls(
            name=name,
            model_type='random_forest',
            feature_set=feature_set,
            base_params={'n_estimators': 100, 'max_depth': 3},
            description="Random Forest Regressor"
        )

    @classmethod
    def XGBoost(cls, name: str, feature_set: str = 'all'):
        return cls(
            name=name,
            model_type='xgboost',
            feature_set=feature_set,
            base_params={'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
            description="XGBoost Regressor"
        )

    @classmethod
    def MLP(cls, name: str, feature_set: str = 'all', n_hidden_layers: int = 3):
        return cls(
            name=name,
            model_type='mlp',
            feature_set=feature_set,
            base_params={'shuffle': False, 'verbose': False},
            architecture={'n_hidden_layers': n_hidden_layers},
            description="MLP Regressor"
        )