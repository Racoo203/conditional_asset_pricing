import numpy as np
import pandas as pd

from src.models.tuners import OptunaTuner
from src.models.validator import RollingWindowValidator
from src.models.factory import ModelFactory
from src.models.spec import ModelSpec
from src.models.search_spaces import HuberSearchSpace


def make_panel():
    dates = pd.date_range("2020-01-01", periods=6, freq="M")
    rows = []

    for d in dates:
        for _ in range(4):
            rows.append({
                "date": d,
                "x1": np.random.randn(),
                "x2": np.random.randn(),
                "target_ret_excess": np.random.randn()
            })

    return pd.DataFrame(rows)


def test_optuna_tuner_returns_params(tmp_path):
    df = make_panel()
    spec = ModelSpec.OLS_Huber(name="optuna_test")
    validator = RollingWindowValidator(n_splits=2)

    tuner = OptunaTuner(
        storage=f"sqlite:///{tmp_path}/optuna.db"
    )

    params, meta = tuner.optimize(
        spec=spec,
        build_fn=ModelFactory,
        validator=validator,
        df_dev=df,
        search_space=HuberSearchSpace(),
        n_trials=3,
        study_name="test_study"
    )

    assert isinstance(params, dict)
    assert "alpha" in params
    assert meta["method"] == "optuna"