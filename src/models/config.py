from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class ModelConfig:
    name: str
    model_type: str  # 'huber', 'sgd_huber'
    feature_set: str # 'all', 'ff3' (Size, Value, Mom)
    params: Dict[str, Any] = field(default_factory=dict)
    use_optuna: bool = False
    optuna_trials: int = 20
    
    # Specific attributes for reproducibility
    description: Optional[str] = None