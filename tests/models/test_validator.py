import numpy as np
import pandas as pd

from src.models.validator import RollingWindowValidator
from src.models.factory import ModelFactory
from src.models.spec import ModelSpec


def make_panel(n_dates=6, n_assets=5):
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="ME")
    rows = []

    for d in dates:
        for i in range(n_assets):
            rows.append({
                "date": d,
                "x1": np.random.randn(),
                "x2": np.random.randn(),
                "target_ret_excess": np.random.randn()
            })

    return pd.DataFrame(rows)


def test_rolling_validator_runs():
    df = make_panel()
    spec = ModelSpec.OLS_Huber(name="test")
    validator = RollingWindowValidator(n_splits=2)

    def build_fn(params):
        return ModelFactory.create(spec, params)

    result = validator.score(
        build_fn=build_fn,
        params=spec.base_params,
        df_dev=df,
        feature_set="all",
        target_col="target_ret_excess",
        date_col="date"
    )

    assert "mean_r2" in result
    assert "mean_rmse" in result
    assert result["n_folds"] > 0