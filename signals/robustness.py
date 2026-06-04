"""
Robustness tests on the signal side.

Varies:
  - Momentum lookback : 6-1, 9-1, 12-1
  - Residualization window : 24, 36, 48 months
  - Factor model : MKT (market only) vs FF3 (Mkt-RF + SMB + HML)

Saves every signal variant to data/raw/signals/ and prints a summary table.
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from signals.momentum import baseline_momentum, _rolling_residuals

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
SIG_DIR = RAW_DIR / "signals"
SIG_DIR.mkdir(parents=True, exist_ok=True)

LOOKBACKS = [6, 9, 12] # with skip=1 throughout
RESID_WINDOWS = [24, 36, 48]
FACTOR_MODELS = {
    "MKT": ["Mkt-RF"],
    "FF3": ["Mkt-RF", "SMB", "HML"],
}


# Residual momentum for one parameter combination.
def _resid_mom_variant(
    ind: pd.DataFrame,
    factors: pd.DataFrame,
    lookback: int,
    resid_window: int,
    factor_cols: list[str],
) -> pd.DataFrame:
    resid = _rolling_residuals(ind, factors[factor_cols], window=resid_window)
    return baseline_momentum(resid, lookback=lookback, skip=1)


# Compute all signal variants. Returns a dict keyed by variant name.
def run_all(verbose: bool = True) -> dict[str, pd.DataFrame]:
    ind = pd.read_parquet(RAW_DIR / "industry_returns.parquet")
    ff3 = pd.read_parquet(RAW_DIR / "ff3_factors.parquet")

    results = {}
    summary_rows = []

    # Baseline momentum variants (no residualization)
    for lb in LOOKBACKS:
        name = f"baseline_L{lb}"
        sig = baseline_momentum(ind, lookback=lb, skip=1)
        sig.to_parquet(SIG_DIR / f"{name}.parquet")
        results[name] = sig
        summary_rows.append(_summarize(name, sig))

    # Residual momentum variants
    total = len(LOOKBACKS) * len(RESID_WINDOWS) * len(FACTOR_MODELS)
    done = 0
    for lb, rw, (fm_name, fcols) in product(LOOKBACKS, RESID_WINDOWS, FACTOR_MODELS.items()):
        name = f"residual_L{lb}_W{rw}_{fm_name}"
        if verbose:
            done += 1
            print(f"  [{done}/{total}] {name} ...", end="\r")
        sig = _resid_mom_variant(ind, ff3, lookback=lb, resid_window=rw, factor_cols=fcols)
        sig.to_parquet(SIG_DIR / f"{name}.parquet")
        results[name] = sig
        summary_rows.append(_summarize(name, sig))

    if verbose:
        print() # newline after \r

    summary = pd.DataFrame(summary_rows).set_index("variant")
    if verbose:
        print("\nSignal robustness summary:")
        print(summary.to_string())

    summary.to_csv(SIG_DIR / "summary.csv")
    return results


def _summarize(name: str, sig: pd.DataFrame) -> dict:
    valid = sig.stack().dropna()
    first_valid = sig.dropna(how="all").index[0].date() if not sig.dropna(how="all").empty else None
    return {
        "variant": name,
        "first_valid": first_valid,
        "nan_pct": f"{sig.isna().mean().mean():.1%}",
        "mean": round(valid.mean(), 3),
        "std": round(valid.std(), 3),
        "p10": round(valid.quantile(0.10), 3),
        "p90": round(valid.quantile(0.90), 3),
    }


if __name__ == "__main__":
    print("Running signal robustness variants...")
    run_all(verbose=True)
    print(f"\nAll variants saved to {SIG_DIR}")
