"""
gsd.forecasting — the two estimators, side by side.

    OLS    ordinary least squares on lagged values        (the review baseline)
           univariate AR(p), multivariate VAR(p)
    SPTLS  the non-abelian one-step fit on the kinematic   (the programme's
           embedding q(t)=(z,Dz,D2z); M-hat read by polar   estimator)
           decomposition, with full-Mahalanobis whitening

Design note. The OLS result object deliberately mirrors the layout of
`statsmodels`' regression summary (coefficient table with standard errors,
t-statistics, p-values, confidence intervals, plus R^2 / AIC / BIC /
Durbin-Watson). We reimplement it in plain numpy so the library carries no
heavy dependency; we gratefully acknowledge `statsmodels` (Seabold &
Perktold, 2010) as the reference design for that summary table. p-values use
a large-sample normal approximation (the t-statistics are exact).

numpy only.
"""
from __future__ import annotations
import numpy as np

from .embedding import kinematic_embedding, stack_embedding
from . import metrics as _m
from . import algebra as _alg

__all__ = ["OLS", "SPTLS", "OLSResult", "SPTLSResult"]


# ----------------------------------------------------------------------
#  OLS  —  AR(p) / VAR(p)
# ----------------------------------------------------------------------
def _design_ar(y, p, trend="c"):
    y = np.asarray(y, float).ravel()
    T = len(y)
    rows = T - p
    cols = []
    names = []
    if "c" in trend:
        cols.append(np.ones(rows)); names.append("const")
    for k in range(1, p + 1):
        cols.append(y[p - k:T - k]); names.append(f"L{k}.y")
    X = np.column_stack(cols)
    yv = y[p:]
    return X, yv, names


class OLSResult:
    """Fitted OLS model. Attributes echo statsmodels naming."""

    def __init__(self, params, X, y, names, kind, extra=None):
        self.params = np.asarray(params, float)
        self._X = X
        self._y = y
        self.names = names
        self.kind = kind
        self.extra = extra or {}
        self.fittedvalues = X @ self.params
        self.resid = y - self.fittedvalues
        n, k = X.shape
        self.nobs = n
        self.df_resid = n - k
        rss = float(self.resid @ self.resid)
        self.ssr = rss
        sigma2 = rss / max(self.df_resid, 1)
        XtX_inv = np.linalg.pinv(X.T @ X)
        self.cov_params = sigma2 * XtX_inv
        self.bse = np.sqrt(np.clip(np.diag(self.cov_params), 0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            self.tvalues = self.params / self.bse
        self.pvalues = np.array([2.0 * _m.norm_sf(abs(t)) for t in self.tvalues])
        self.rsquared = _m.r2(self.fittedvalues, y)
        tss = np.sum((y - y.mean()) ** 2)
        self.rsquared_adj = 1.0 - (1.0 - self.rsquared) * (n - 1) / max(n - k, 1)
        self.aic = _m.aic(rss, n, k)
        self.bic = _m.bic(rss, n, k)
        self.dw = _m.durbin_watson(self.resid)

    def conf_int(self, alpha=0.05):
        z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _zcrit(alpha)
        lo = self.params - z * self.bse
        hi = self.params + z * self.bse
        return np.column_stack([lo, hi])

    def predict(self, X=None):
        return self.fittedvalues if X is None else np.asarray(X, float) @ self.params

    def forecast_next(self, history):
        """One-step-ahead AR forecast from the tail of a 1-D history."""
        h = np.asarray(history, float).ravel()
        p = self.extra.get("p", sum(1 for nm in self.names if nm.startswith("L")))
        val = 0.0
        idx = 0
        if "const" in self.names:
            val += self.params[0]; idx = 1
        for k in range(1, p + 1):
            val += self.params[idx] * h[-k]; idx += 1
        return float(val)

    def summary(self):
        ci = self.conf_int()
        lines = []
        lines.append("=" * 74)
        lines.append(f" OLS Regression Results   ({self.kind})")
        lines.append("=" * 74)
        lines.append(f" No. Observations: {self.nobs:>6d}     R-squared:      {self.rsquared:>8.4f}")
        lines.append(f" Df Residuals:     {self.df_resid:>6d}     Adj. R-squared: {self.rsquared_adj:>8.4f}")
        lines.append(f" AIC:           {self.aic:>9.3f}     BIC:         {self.bic:>9.3f}")
        lines.append(f" Durbin-Watson: {self.dw:>9.3f}")
        lines.append("-" * 74)
        lines.append(f"{'':>10}{'coef':>10}{'std err':>10}{'t':>9}{'P>|t|':>9}{'[0.025':>9}{'0.975]':>9}")
        lines.append("-" * 74)
        for i, nm in enumerate(self.names):
            lines.append(f"{nm:>10}{self.params[i]:>10.4f}{self.bse[i]:>10.4f}"
                         f"{self.tvalues[i]:>9.3f}{self.pvalues[i]:>9.3f}"
                         f"{ci[i,0]:>9.4f}{ci[i,1]:>9.4f}")
        lines.append("=" * 74)
        lines.append(" p-values: large-sample normal approximation (numpy-only build).")
        return "\n".join(lines)

    def __repr__(self):
        return self.summary()


class OLS:
    """Ordinary least squares on lagged values.
    Univariate -> AR(p); multivariate (2-D y) -> VAR(p) (one equation per series)."""

    def __init__(self, p=1, trend="c"):
        self.p = p
        self.trend = trend

    def fit(self, y):
        y = np.asarray(y, float)
        if y.ndim == 1:
            X, yv, names = _design_ar(y, self.p, self.trend)
            params, *_ = np.linalg.lstsq(X, yv, rcond=None)
            return OLSResult(params, X, yv, names, kind=f"AR({self.p})",
                             extra={"p": self.p, "trend": self.trend})
        # VAR(p): regress each series on its own + others' lags
        return _fit_var(y, self.p, self.trend)


class VARResult:
    """Container of per-equation OLSResult objects for a VAR(p)."""

    def __init__(self, eqs, names):
        self.equations = eqs        # list of OLSResult
        self.names = names          # series names

    @property
    def rsquared(self):
        return {nm: eq.rsquared for nm, eq in zip(self.names, self.equations)}

    def forecast_next(self, history):
        """One-step-ahead VAR forecast (d-vector) from a (T, d) history."""
        H = np.atleast_2d(np.asarray(history, float))
        out = np.zeros(len(self.equations))
        for i, eq in enumerate(self.equations):
            val = 0.0
            for idx, nm in enumerate(eq.names):
                if nm == "const":
                    val += eq.params[idx]
                else:                       # name like "Lk.yj"
                    k = int(nm[1:nm.index(".")])
                    j = int(nm[nm.index("y") + 1:])
                    val += eq.params[idx] * H[-k, j]
            out[i] = val
        return out

    def summary(self):
        out = [f"VAR({self.equations[0].extra.get('p','?')}) — {len(self.equations)} equations"]
        for nm, eq in zip(self.names, self.equations):
            out.append(f"\n### equation for {nm}")
            out.append(eq.summary())
        return "\n".join(out)

    def __repr__(self):
        return self.summary()


def _fit_var(Y, p, trend):
    Y = np.atleast_2d(Y)
    T, d = Y.shape
    names = [f"y{j}" for j in range(d)]
    rows = T - p
    cols = []; cnames = []
    if "c" in trend:
        cols.append(np.ones(rows)); cnames.append("const")
    for k in range(1, p + 1):
        for j in range(d):
            cols.append(Y[p - k:T - k, j]); cnames.append(f"L{k}.y{j}")
    X = np.column_stack(cols)
    eqs = []
    for i in range(d):
        yv = Y[p:, i]
        params, *_ = np.linalg.lstsq(X, yv, rcond=None)
        r = OLSResult(params, X, yv, cnames, kind=f"VAR({p}) eq y{i}",
                      extra={"p": p})
        eqs.append(r)
    return VARResult(eqs, names)


def _zcrit(alpha):
    # crude inverse-normal for two-sided alpha (Acklam approximation)
    import math
    p = 1 - alpha / 2
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    dd = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00]
    plow = 0.02425; phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)
    if p <= phigh:
        q = p - 0.5; r = q*q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((dd[0]*q+dd[1])*q+dd[2])*q+dd[3])*q+1)


# ----------------------------------------------------------------------
#  SPTLS  —  the non-abelian one-step fit
# ----------------------------------------------------------------------
def _inv_sqrt_psd(C, eps=1e-12):
    w, V = np.linalg.eigh(0.5 * (C + C.T))
    w = np.clip(w, eps, None)
    return V @ np.diag(1.0 / np.sqrt(w)) @ V.T, V @ np.diag(np.sqrt(w)) @ V.T


class SPTLSResult:
    """Fitted SPTLS one-step operator on the kinematic embedding."""

    def __init__(self, M, C0, q0, q1, d, h, whiten):
        self.M = M                  # operator in original coordinates (3d x 3d)
        self.C0 = C0                # embedding Gram
        self.d = d                  # number of series
        self.h = h
        self.whiten = whiten
        self._q0 = q0
        self._q1 = q1
        pred = (M @ q0.T).T
        self.fitted_jet = pred
        # whitened operator for the pristine polar read
        Wn, Wp = _inv_sqrt_psd(C0)
        self.M_white = Wn @ M @ Wp
        # per-series z-channel R^2 (the forecasting-relevant number)
        self.rsquared = {}
        for i in range(d):
            zc = 3 * i
            self.rsquared[f"y{i}"] = _m.r2(pred[:, zc], q1[:, zc])
        self.resid_jet = q1 - pred

    @property
    def rsquared_mean(self):
        return float(np.mean(list(self.rsquared.values())))

    def predict_jet(self, q):
        return (self.M @ np.atleast_2d(q).T).T

    def forecast_next(self, history, h=None):
        """One-step-ahead level forecast from the tail of a history.
        Univariate: 1-D history. Multivariate: (T, d) history."""
        h = self.h if h is None else h
        hist = np.asarray(history, float)
        if self.d == 1:
            from .embedding import kinematic_embedding
            Q = kinematic_embedding(hist, h)
            q_last = Q[-1]
            return float((self.M @ q_last)[0])
        from .embedding import stack_embedding
        Q = stack_embedding(hist, h)
        jet = self.M @ Q[-1]
        return np.array([jet[3 * i] for i in range(self.d)])

    def report(self):
        from .diagnostics import SPTLSReport
        return SPTLSReport(self)

    def summary(self):
        return self.report().summary()

    def __repr__(self):
        return self.summary()


def _quadratic_features(q0, d):
    """Augment the (stacked) jet with the degree-2 (quadratic) closure of the
    level channels: [1, full jet, {z_i z_j : i<=j}].  z_i is column 3i.

    Note on terminology: these are *polynomial degree-2* terms (symmetric
    products z_i z_j), NOT the Clifford *grade-2* bivectors. The grade-2
    (bivector / rotational) content of the dynamics is already carried by the
    antisymmetric part / rotor of the linear operator M and is read off every
    linear fit. This dictionary is a separate axis — nonlinearity — added only
    when the dynamics is not linear (e.g. a logistic or quadratic-coupled map).
    """
    z = q0[:, ::3]                                   # (n, d) level channels
    cols = [np.ones(len(q0))]
    cols += [q0[:, k] for k in range(q0.shape[1])]   # the full linear jet
    for i in range(d):
        for j in range(i, d):
            cols.append(z[:, i] * z[:, j])           # symmetric degree-2 terms
    return np.column_stack(cols)


class SPTLSQuadraticResult:
    """SPTLS with the degree-2 (quadratic) dictionary — a forecasting fit for
    nonlinear dynamics that the linear operator cannot represent. Exposes
    .rsquared / .forecast_next; the polar reading (.report) is defined only for
    the linear operator, so it is not provided here."""

    def __init__(self, B, q0, q1, d, h):
        self.B = B                  # (d, n_features) coefficients, per z-channel
        self.d = d
        self.h = h
        self._q0 = q0
        X = _quadratic_features(q0, d)
        self.rsquared = {}
        self.fitted_z = np.zeros((len(q0), d))
        for i in range(d):
            pred = X @ B[i]
            self.fitted_z[:, i] = pred
            self.rsquared[f"y{i}"] = _m.r2(pred, q1[:, 3 * i])

    @property
    def rsquared_mean(self):
        return float(np.mean(list(self.rsquared.values())))

    def forecast_next(self, history, h=None):
        h = self.h if h is None else h
        hist = np.asarray(history, float)
        Q = kinematic_embedding(hist, h) if self.d == 1 else stack_embedding(hist, h)
        feat = _quadratic_features(Q[-1:], self.d)[0]
        out = np.array([self.B[i] @ feat for i in range(self.d)])
        return float(out[0]) if self.d == 1 else out


class SPTLS:
    """The non-abelian one-step least-squares fit of the kinematic embedding.

    Parameters
    ----------
    h : float          finite-difference step for the jet (default 1.0).
    whiten : str       'mahalanobis' (default) reads the rotor in the
                       isotropic geometry C0^{-1/2}; 'none' reads it raw.
    dictionary : str   'linear' (default) — the one-step operator M, with the
                       full polar/diagnostic reading; 'quadratic' — augment
                       with the polynomial degree-2 closure of the levels, for
                       nonlinear dynamics a linear map cannot represent
                       (forecasting only; no polar reading). 'grade2' is
                       accepted as a deprecated alias for 'quadratic'.

    Terminology: 'quadratic' is the *polynomial degree* (z_i z_j terms), a
    different axis from the Clifford *grade*. The grade-2 (bivector) /
    rotational content lives in the linear operator's rotor and is read off
    every linear fit; it is always present, never opt-in.
    """

    def __init__(self, h=1.0, whiten="mahalanobis", dictionary="linear"):
        self.h = h
        self.whiten = whiten
        self.dictionary = "quadratic" if dictionary == "grade2" else dictionary

    def fit(self, y):
        y = np.asarray(y, float)
        if y.ndim == 1:
            Q = kinematic_embedding(y, self.h); d = 1
        else:
            Q = stack_embedding(y, self.h); d = y.shape[1]
        q0, q1 = Q[:-1], Q[1:]
        if self.dictionary == "quadratic":
            X = _quadratic_features(q0, d)
            B = np.array([np.linalg.lstsq(X, q1[:, 3 * i], rcond=None)[0]
                          for i in range(d)])
            return SPTLSQuadraticResult(B, q0, q1, d, self.h)
        C0 = q0.T @ q0
        C1 = q1.T @ q0
        M = C1 @ np.linalg.pinv(C0)
        return SPTLSResult(M, C0, q0, q1, d, self.h, self.whiten)
