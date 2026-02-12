#src/evaluation/metrics.py
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

def regression_metrics(y_true, y_pred):
    """
    Returns a dict of standard regression metrics.
    Note: R2_OOS in finance is 1 - (MSE_model / MSE_zero), 
    but sklearn's r2_score is 1 - (MSE_model / MSE_mean).
    
    For excess returns, MSE_zero is often the tougher benchmark.
    Both are provided, but when speaking of R2 Out-Of-Sample,
    it refers to the financial application.
    """
    mse_model = mean_squared_error(y_true, y_pred)
    mse_zero = np.mean(np.square(y_true))
    
    return {
        'rmse': np.sqrt(mse_model),
        'r2_sklearn': r2_score(y_true, y_pred),
        'r2_oos': 1 - (mse_model / mse_zero)
    }