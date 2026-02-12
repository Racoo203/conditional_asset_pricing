import numpy as np
import pandas as pd

from src.models.factory import ModelFactory
from src.models.spec import ModelSpec


def make_toy_data(n=20, p=3):
    X = pd.DataFrame(np.random.randn(n, p), columns=[f"x{i}" for i in range(p)])
    y = X.sum(axis=1) + np.random.randn(n) * 0.01
    return X, y


def test_huber_factory_builds_and_runs():
    spec = ModelSpec.OLS_Huber(name="test")
    X, y = make_toy_data()

    model = ModelFactory.create(spec, spec.base_params)
    model.fit(X, y)
    preds = model.predict(X)

    assert preds.shape[0] == len(y)


def test_mlp_factory_uses_architecture():
    spec = ModelSpec.MLP(name="mlp", n_hidden_layers=2)
    X, y = make_toy_data()

    model = ModelFactory.create(spec, spec.base_params)
    model.fit(X, y)

    assert len(model.hidden_layer_sizes) == 2
