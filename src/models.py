# src/models.py
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def train_ridge_time_series(X, y, n_splits=5, alpha=1.0):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    mses = []
    model = None
    for train_idx, test_idx in tscv.split(Xs):
        X_train, X_test = Xs[train_idx], Xs[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = Ridge(alpha=alpha).fit(X_train, y_train)
        preds = model.predict(X_test)
        mses.append(mean_squared_error(y_test, preds))
    # final model fit on full data
    final_model = Ridge(alpha=alpha).fit(Xs, y)
    return {'model': final_model, 'scaler': scaler}, mses
