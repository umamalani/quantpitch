"""
Momentum signals for the 49-industry residual momentum strategy.

baseline_momentum : 12-1 cumulative return per industry
residual_momentum : same window on FF3-residualized returns
"""

from pathlib import Path
import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


# Cumulative return from t-lookback to t-(skip+1), inclusive.
# Default 12-1: skips t-1, accumulates t-2 through t-12 (11 months).
# Returns percent, same shape as ind.
def baseline_momentum(
    ind: pd.DataFrame,
    lookback: int = 12,
    skip: int = 1,
) -> pd.DataFrame:
    gross = 1 + ind / 100
    # shift(skip+1) so row t aligns to return at t-(skip+1)
    gross_shifted = gross.shift(skip + 1)
    window = lookback - skip  # 11 for 12-1
    cum_log = np.log(gross_shifted).rolling(window).sum()
    return (np.exp(cum_log) - 1) * 100


# At each month t, regress each industry's trailing 'window' returns on
# Mkt-RF, SMB, HML using numpy lstsq. Returns the residual at month t.
def _rolling_residuals(
    ind: pd.DataFrame,
    factors: pd.DataFrame,
    window: int = 36,
) -> pd.DataFrame:
    factor_cols = [c for c in ["Mkt-RF", "SMB", "HML"] if c in factors.columns]
    shared = ind.index.intersection(factors.index)
    Y = ind.loc[shared].values.astype(float) # (T, N)
    F = factors.loc[shared, factor_cols].values.astype(float) # (T, 3)

    T, N = Y.shape
    resid_arr = np.full((T, N), np.nan)

    for t in range(window - 1, T):
        sl = slice(t - window + 1, t + 1)
        X_win = np.column_stack([np.ones(window), F[sl]]) # (window, 4)
        Y_win = Y[sl] # (window, N)

        if not np.all(np.isfinite(X_win)):
            continue

        # Industries with complete data in this window — solve jointly
        complete = np.all(np.isfinite(Y_win), axis=0)
        if complete.any():
            B, _, _, _ = np.linalg.lstsq(X_win, Y_win[:, complete], rcond=None)
            resid_arr[t, complete] = Y_win[-1, complete] - X_win[-1] @ B

        # Industries with some within-window NaN but valid at t — solve individually
        for j in np.where(~complete & np.isfinite(Y_win[-1]))[0]:
            y_j = Y_win[:, j]
            valid = np.isfinite(y_j)
            if valid.sum() < window // 2:
                continue
            B_j, _, _, _ = np.linalg.lstsq(X_win[valid], y_j[valid], rcond=None)
            resid_arr[t, j] = y_j[-1] - X_win[-1] @ B_j

    return pd.DataFrame(resid_arr, index=shared, columns=ind.loc[shared].columns)


# 12-1 momentum on FF3-residualized returns.
# resid_window: trailing months used to estimate each rolling regression.
def residual_momentum(
    ind: pd.DataFrame,
    factors: pd.DataFrame,
    lookback: int = 12,
    skip: int = 1,
    resid_window: int = 36,
) -> pd.DataFrame:
    resid = _rolling_residuals(ind, factors, window=resid_window)
    return baseline_momentum(resid, lookback=lookback, skip=skip)


# Load raw data and compute both baseline and residual momentum signals.
def load_and_compute(
    lookback: int = 12,
    skip: int = 1,
    resid_window: int = 36,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ind = pd.read_parquet(RAW_DIR / "industry_returns.parquet")
    ff3 = pd.read_parquet(RAW_DIR / "ff3_factors.parquet")
    base = baseline_momentum(ind, lookback=lookback, skip=skip)
    resid = residual_momentum(ind, ff3, lookback=lookback, skip=skip, resid_window=resid_window)
    return base, resid


if __name__ == "__main__":
    print("Computing momentum signals (this takes ~30s for rolling regressions)...")
    base, resid = load_and_compute()

    base.to_parquet(RAW_DIR / "baseline_momentum.parquet")
    resid.to_parquet(RAW_DIR / "residual_momentum.parquet")

    first_valid = base.dropna(how="all").index[0].date()
    print(f"Baseline momentum : {base.shape}  first valid: {first_valid}")
    print(f"Residual momentum : {resid.shape}  NaN rate: {resid.isna().mean().mean():.1%}")
    print(f"Saved to {RAW_DIR}")
