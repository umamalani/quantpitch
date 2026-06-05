"""
Fetches Ken French 49-industry returns, FF3 factors, and the momentum
factor (UMD) directly from the French Data Library CSV zips. Aligns them
to a common monthly DatetimeIndex and saves parquet files to data/raw/.

This version downloads CSVs directly via urllib so it has no third-party
dependencies beyond pandas. Earlier versions used pandas_datareader, which
has Python 3.14 compatibility issues.
"""

from pathlib import Path
import io
import re
import zipfile
import urllib.request

import pandas as pd

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# CSV zip URLs from Ken French's data library
_URLS = {
    "industries": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_CSV.zip",
    "ff3":        "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip",
    "momentum":   "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip",
}

_YYYYMM = re.compile(r"^\d{6},")  # lines starting with YYYYMM, are monthly data rows


# Downloads the zip from `url`, extracts the first CSV inside, returns its text.
def _download_csv(url: str) -> str:
    with urllib.request.urlopen(url) as resp:
        with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
            with zf.open(zf.namelist()[0]) as f:
                return f.read().decode("utf-8", errors="replace")


# Parses a French CSV string into a DataFrame indexed by month-end dates.
# Strategy: find the header line (the line just before the first YYYYMM row),
# then read contiguous YYYYMM rows until we hit a non-data line. This correctly
# stops before any annual-returns section or auxiliary table (e.g. equal-weighted).
def _parse_french_csv(raw: str) -> pd.DataFrame:
    lines = raw.split("\n")
    first_data = next(i for i, l in enumerate(lines) if _YYYYMM.match(l))

    # The header is the most recent non-blank line above first_data
    header_idx = first_data - 1
    while header_idx > 0 and not lines[header_idx].strip():
        header_idx -= 1
    header_line = lines[header_idx]

    # Read contiguous monthly rows starting at first_data
    end = first_data
    while end < len(lines) and _YYYYMM.match(lines[end]):
        end += 1

    data_block = header_line + "\n" + "\n".join(lines[first_data:end])
    df = pd.read_csv(io.StringIO(data_block))

    # First column is the YYYYMM date — pandas will name it 'Unnamed: 0' because
    # the header starts with a comma. Rename and convert.
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.set_index("date")
    df.columns = df.columns.str.strip()

    return df


# Loads 49-industry value-weighted monthly returns from Ken French.
def load_industry_returns(start: str = "1963-07", end: str = None) -> pd.DataFrame:
    raw = _download_csv(_URLS["industries"])
    ind = _parse_french_csv(raw)
    ind = ind.replace([-99.99, -999.0], float("nan"))
    return _slice(ind, start, end)


# Loads FF3 factors (Mkt-RF, SMB, HML, RF) from Ken French.
def load_ff3_factors(start: str = "1963-07", end: str = None) -> pd.DataFrame:
    raw = _download_csv(_URLS["ff3"])
    ff3 = _parse_french_csv(raw)
    return _slice(ff3, start, end)


# Loads the momentum factor (renamed to 'UMD' for clarity).
def load_momentum_factor(start: str = "1963-07", end: str = None) -> pd.DataFrame:
    raw = _download_csv(_URLS["momentum"])
    mom = _parse_french_csv(raw)
    mom = mom.rename(columns={mom.columns[0]: "UMD"})
    return _slice(mom, start, end)


def _slice(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_dt = pd.Timestamp(start) + pd.offsets.MonthEnd(0)
    end_dt = (pd.Timestamp(end) + pd.offsets.MonthEnd(0)) if end else df.index[-1]
    return df.loc[start_dt:end_dt]


# Returns (industries, factors) aligned to a shared monthly DatetimeIndex.
# `factors` includes Mkt-RF, SMB, HML, RF, and UMD.
def build_panel(start: str = "1963-07", end: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ind = load_industry_returns(start=start, end=end)
    ff3 = load_ff3_factors(start=start, end=end)
    mom = load_momentum_factor(start=start, end=end)

    factors = ff3.join(mom, how="inner")
    shared = ind.index.intersection(factors.index)
    return ind.loc[shared], factors.loc[shared]


def save_raw(start: str = "1963-07", end: str = None) -> None:
    ind, factors = build_panel(start=start, end=end)

    ind_path = RAW_DIR / "industry_returns.parquet"
    fac_path = RAW_DIR / "ff3_factors.parquet"  # keep name for backward compat; now includes UMD

    ind.to_parquet(ind_path)
    factors.to_parquet(fac_path)

    print(f"Saved {len(ind)} months of industry returns  -> {ind_path}")
    print(f"Saved {len(factors)} months of factors          -> {fac_path}")
    print(f"Date range: {ind.index[0].date()} to {ind.index[-1].date()}")
    print(f"Industries: {ind.shape[1]}  |  Factor columns: {list(factors.columns)}")


if __name__ == "__main__":
    save_raw()