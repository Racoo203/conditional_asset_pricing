from sklearn.linear_model import HuberRegressor, SGDRegressor, LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from src.models.spec import ModelSpec

class ModelFactory:
    """
    Deterministic model builder.
    """

    @staticmethod
    def create(spec: ModelSpec, params: dict):
        model_type = spec.model_type

        if model_type == 'huber':
            return HuberRegressor(max_iter=500, **params)

        elif model_type == 'sgd_huber':
            return SGDRegressor(
                loss='huber',
                penalty='elasticnet',
                max_iter=1000,
                **params
            )

        elif model_type == 'pcr':
            return Pipeline([
                ('pca', PCA(n_components=params['n_components'])),
                ('reg', LinearRegression())
            ])

        elif model_type == 'random_forest':
            return RandomForestRegressor(n_jobs=-1, **params)

        elif model_type == 'xgboost':
            return XGBRegressor(
                objective='reg:squarederror',
                n_jobs=-1,
                **params
            )

        elif model_type == 'mlp':
            n_layers = spec.architecture['n_hidden_layers']
            layers = tuple(max(4, 2 ** (6 - i)) for i in range(n_layers))

            return MLPRegressor(
                hidden_layer_sizes=layers,
                early_stopping=True,
                max_iter=1000,
                **params
            )

        else:
            raise ValueError(f"Unknown model_type: {model_type}")
