import optuna

class BaseSearchSpace:
    def suggest(self, trial: optuna.Trial) -> dict:
        raise NotImplementedError


class HuberSearchSpace(BaseSearchSpace):
    def suggest(self, trial):
        return {
            'epsilon': trial.suggest_float('epsilon', 1.0, 1.9),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
        }


class SGDHuberSearchSpace(BaseSearchSpace):
    def suggest(self, trial):
        return {
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
        }


class PCRSearchSpace(BaseSearchSpace):
    def suggest(self, trial):
        return {
            'n_components': trial.suggest_int('n_components', 1, 30)
        }


class RandomForestSearchSpace(BaseSearchSpace):
    def suggest(self, trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 50),
        }


class XGBoostSearchSpace(BaseSearchSpace):
    def suggest(self, trial):
        return {
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 2, 6),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        }


class MLPSearchSpace(BaseSearchSpace):
    def suggest(self, trial):
        return {
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'learning_rate_init': trial.suggest_float(
                'learning_rate_init', 1e-4, 1e-2, log=True
            ),
        }