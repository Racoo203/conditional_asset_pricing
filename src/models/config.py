from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class ModelConfig:
    name: str
    model_type: str   # 'huber', 'sgd_huber'
    feature_set: str  # 'all', 'ff3'
    params: Dict[str, Any] = field(default_factory=dict)
    use_optuna: bool = False
    optuna_trials: int = 20
    description: Optional[str] = None

    @classmethod
    def OLS_Huber(cls, name: str, feature_set: str = 'all', epsilon: float = 1.35):
        """Creates a standard OLS-Huber configuration."""
        return cls(
            name=name,
            model_type='huber',
            feature_set=feature_set,
            params={'epsilon': epsilon},
            use_optuna=False
        )

    @classmethod
    def ElasticNet_Huber(cls, name: str, feature_set: str = 'all', trials: int = 20):
        """Creates a Tunable ElasticNet-Huber configuration."""
        return cls(
            name=name,
            model_type='sgd_huber',
            feature_set=feature_set,
            params={'alpha': 0.0001, 'l1_ratio': 0.15}, # Defaults/Starting points
            use_optuna=True,
            optuna_trials=trials
        )