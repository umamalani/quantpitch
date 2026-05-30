"""
Fetches Ken French 49-industry returns and FF3 factors from the French Data Library,
aligns them to a common monthly DatetimeIndex, and saves parquet files to data/raw/.
"""

from pathlib import Path
import pandas as pd
import pandas_datareader.data as web

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# French Data Library dataset names
_IND49_DATASET = "49_Industry_Portfolios"
_FF3_DATASET = "F-F_Research_Data_Factors"


def _to_monthly_index(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a French-library index (YYYY-MM or YYYYMM) to month-end DatetimeIndex."""
    df.index = pd.to_datetime(df.index.astype(str), format="mixed") + pd.offsets.MonthEnd(0)
    df.index.name = "date"
    return df


def load_industry_returns(start: str = "1963-07", end: str = None) -> pd.DataFrame:
    """
    Returns a DataFrame of 49-industry value-weighted monthly returns in percent.
    Rows: month-end dates. Columns: industry names (stripped of whitespace).
    """
    raw = web.DataReader(_IND49_DATASET, "famafrench", start="1926-01")[0]
    ind = _to_monthly_index(raw)
    ind.columns = ind.columns.str.strip()

    # French fills missing observations with -99.99 or -999
    ind = ind.replace([-99.99, -999.0], float("nan"))

    start_dt = pd.Timestamp(start) + pd.offsets.MonthEnd(0)
    end_dt = (pd.Timestamp(end) + pd.offsets.MonthEnd(0)) if end else ind.index[-1]
    return ind.loc[start_dt:end_dt]


def load_ff3_factors(start: str = "1963-07", end: str = None) -> pd.DataFrame:
    """
    Returns a DataFrame of FF3 factors (Mkt-RF, SMB, HML) and RF in percent.
    Rows: month-end dates.
    """
    raw = web.DataReader(_FF3_DATASET, "famafrench", start="1926-01")[0]
    ff3 = _to_monthly_index(raw)
    ff3.columns = ff3.columns.str.strip()

    start_dt = pd.Timestamp(start) + pd.offsets.MonthEnd(0)
    end_dt = (pd.Timestamp(end) + pd.offsets.MonthEnd(0)) if end else ff3.index[-1]
    return ff3.loc[start_dt:end_dt]


def build_panel(start: str = "1963-07", end: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligns industry returns and FF3 factors to the same month-end DatetimeIndex.
    Returns (industries, factors) — both in percent, same index.
    """
    ind = load_industry_returns(start=start, end=end)
    ff3 = load_ff3_factors(start=start, end=end)

    # Intersect on date index so both panels are perfectly aligned
    shared_index = ind.index.intersection(ff3.index)
    ind = ind.loc[shared_index]
    ff3 = ff3.loc[shared_index]

    return ind, ff3


def save_raw(start: str = "1963-07", end: str = None) -> None:
    ind, ff3 = build_panel(start=start, end=end)

    ind_path = RAW_DIR / "industry_returns.parquet"
    ff3_path = RAW_DIR / "ff3_factors.parquet"

    ind.to_parquet(ind_path)
    ff3.to_parquet(ff3_path)

    print(f"Saved {len(ind)} months of industry returns  -> {ind_path}")
    print(f"Saved {len(ff3)} months of FF3 factors        -> {ff3_path}")
    print(f"Date range: {ind.index[0].date()} to {ind.index[-1].date()}")
    print(f"Industries: {ind.shape[1]}  |  FF3 columns: {list(ff3.columns)}")


if __name__ == "__main__":
    save_raw()
