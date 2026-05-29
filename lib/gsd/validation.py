"""
gsd.validation — honest out-of-sample evaluation for time series.

Rolling-origin (a.k.a. walk-forward) cross-validation: never trains on the
future. Two schemes — 'expanding' (growing training window) and 'rolling'
(fixed-width sliding window). Works with any estimator exposing
.fit(y) -> result and result.forecast_next(history).

    cv = rolling_origin(SPTLS(), y, start=30, scheme="expanding")
    cv.metrics            # {'rmse':..,'mae':..,'mape':..,'r2':..}
    cv.predictions        # one-step-ahead forecasts aligned to cv.targets

numpy only.
"""
from __future__ import annotations
import numpy as np
from . import metrics as _m

__all__ = ["rolling_origin", "compare", "CVResult"]


class CVResult:
    def __init__(self, predictions, targets, origins, scheme, label=""):
        self.predictions = np.asarray(predictions, float)
        self.targets = np.asarray(targets, float)
        self.origins = np.asarray(origins, int)
        self.scheme = scheme
        self.label = label
        self.metrics = {
            "rmse": _m.rmse(self.predictions, self.targets),
            "mae": _m.mae(self.predictions, self.targets),
            "mape": _m.mape(self.predictions, self.targets),
            "r2": _m.r2(self.predictions, self.targets),
        }

    def summary(self):
        m = self.metrics
        return (f"[{self.label or 'cv'}] {self.scheme} rolling-origin, "
                f"{len(self.targets)} one-step forecasts\n"
                f"  RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  "
                f"MAPE={m['mape']:.2f}%  R2={m['r2']:.4f}")

    def __repr__(self):
        return self.summary()


def rolling_origin(estimator, y, start=None, scheme="expanding", window=None,
                   refit=True, label=""):
    """Walk-forward one-step CV.

    estimator : an unfitted estimator instance (OLS(...), SPTLS(...)).
    y         : 1-D series (univariate) or (T, d) array (multivariate).
    start     : first origin index (default: 60% of the sample).
    scheme    : 'expanding' or 'rolling'.
    window    : training width for the 'rolling' scheme (default = start).
    """
    y = np.asarray(y, float)
    T = y.shape[0]
    if start is None:
        start = max(int(0.6 * T), 10)
    if window is None:
        window = start
    preds, targs, origins = [], [], []
    for t in range(start, T):
        if scheme == "rolling":
            train = y[t - window:t]
        else:
            train = y[:t]
        res = estimator.fit(train)
        yhat = res.forecast_next(train)
        preds.append(yhat)
        targs.append(y[t])
        origins.append(t)
    return CVResult(preds, targs, origins, scheme, label or type(estimator).__name__)


def compare(estimators, y, **kw):
    """Run rolling_origin for several labelled estimators and return a dict
    of CVResult plus a printable ranking table.

        compare({'OLS AR(2)': OLS(2), 'SPTLS': SPTLS()}, y)
    """
    results = {}
    for name, est in estimators.items():
        results[name] = rolling_origin(est, y, label=name, **kw)
    rows = sorted(results.items(), key=lambda kv: kv[1].metrics["rmse"])
    table = [" model".ljust(22) + "RMSE".rjust(10) + "MAE".rjust(10)
             + "MAPE%".rjust(10) + "R2".rjust(10)]
    table.append("-" * 62)
    for name, cv in rows:
        m = cv.metrics
        table.append(f" {name:<21}{m['rmse']:>10.4f}{m['mae']:>10.4f}"
                     f"{m['mape']:>10.2f}{m['r2']:>10.4f}")
    results["_table"] = "\n".join(table)
    return results
