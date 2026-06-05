# quantpitch

Industry momentum is a well-documented anomaly: portfolios that go long industries with high trailing returns and short industries with low trailing returns earn positive risk-adjusted returns on average. This project tests whether residualizing industry returns against the Fama-French 3-factor model (i.e., stripping out market, size, and value exposures before computing momentum) produces a stronger, less factor-dependent signal.

**Headline result**: FF3-residualized industry momentum (12-1 lookback, 36-month residualization window) earns a statistically significant alpha of 4.4% per year (t = 2.15) after controlling for the Fama-French 3 factors plus the standard momentum factor (UMD). Baseline industry momentum, by contrast, does not produce significant alpha (t = 1.31) and is effectively a tilted version of UMD (UMD beta of 1.06, R² of 0.53). Residualization roughly halves both the UMD loading and the model R², indicating that residual industry momentum captures information not already spanned by standard factors.

The finding is robust at the 12-1 lookback: all six tested parameter combinations (varying residualization window and factor model) produce significant alpha (t-stats from 2.15 to 4.25). At shorter lookbacks (6-1 and 9-1), residualization does not rescue the signal––consistent with the view that momentum is fundamentally a 12-month underreaction phenomenon and residualization sharpens but does not manufacture the underlying signal.

## Authors

Uma Malani and Daniel McCance.

## Repository Structure

```
quantpitch/
├── data/
│   ├── load_data.py            # Downloads industry returns + FF3 + UMD from Ken French
│   └── raw/                    # Generated parquet files (gitignored)
├── signals/
│   ├── momentum.py             # Baseline and FF3-residualized momentum signals
│   └── robustness.py           # 21 parameter variants for sensitivity testing
├── notebooks/
│   └── portfolio_prototype.ipynb   # Backtest, alpha test, and robustness sweep
├── results/
│   ├── figures/                # Plots used in the report
│   └── robustness_table.csv    # Full results across all 21 variants
├── requirements.txt
└── README.md
```

## Reproducing the Results

Clone the repo and install dependencies:

```bash
git clone https://github.com/umamalani/quantpitch.git
cd quantpitch
pip install -r requirements.txt
```

Generate the data and signals:

```bash
python -m data.load_data        # downloads industries + FF3 + UMD from Ken French
python -m signals.momentum      # generates baseline and residual momentum signals
python -m signals.robustness    # generates the 21 parameter variants
```

Then open `notebooks/portfolio_prototype.ipynb` and run all cells. This runs the backtest, alpha test, and robustness sweep, and saves figures and tables to `results/`.

## Data

All data is from Ken French's data library:

- **49-Industry Portfolios** (value-weighted monthly returns)
- **Fama-French 3 Research Factors** (Mkt-RF, SMB, HML, RF)
- **Momentum Factor** (UMD)

Sample period: July 1963 through April 2026. The effective backtest window starts later (mid-1960s) once enough history exists for both the residualization window (36 months) and the momentum lookback (12 months).

## Method

1. **Baseline signal**: for each industry at each month-end, compute the cumulative return over months t-12 through t-2 (the standard "12-1" momentum convention, skipping the most recent month to avoid contamination from short-term reversal).

2. **Residual signal**: for each industry, run a rolling 36-month regression of returns on the Fama-French 3 factors. Extract the residuals (industry-specific returns net of factor exposures). Compute the 12-1 cumulative residual return as the signal.

3. **Portfolio construction**: each month, rank industries by signal. Form an equal-weighted long portfolio of the top 5 industries and an equal-weighted short portfolio of the bottom 5. Hold for one month. The strategy return is the long portfolio return minus the short portfolio return.

4. **Evaluation**: annualized Sharpe ratio, maximum drawdown, performance during the March-April 2009 momentum crash, and time-series regression of strategy returns on Mkt-RF + SMB + HML + UMD with Newey-West (HAC, 6 lags) standard errors. Alpha and its t-statistic from this regression are the primary metric.

5. **Robustness**: the same evaluation is repeated across 3 momentum lookbacks (6, 9, 12 months), 3 residualization windows (24, 36, 48 months), and 2 factor models for residualization (CAPM and FF3).

## Limitations

- All testing is in-sample on the full available history.
- Transaction costs are not modeled. Industry-level long-short strategies have moderate turnover, but a realistic cost assumption would reduce the reported returns.
- The analysis is conducted on a single asset class (US equities, aggregated to industries) and a single time series. Cross-validation across countries or asset classes is not performed.
- The strategy is constructed at the industry level, not the individual security level, so industry-level effects cannot be cleanly separated from industry-membership effects within those portfolios.

## Acknowledgments

This project builds on Moskowitz and Grinblatt (1999) for industry momentum and on Blitz, Huij, and Martens (2011) for the residual momentum framework applied to individual stocks. The asset pricing testing methodology follows Fama and MacBeth (1973) and the broader factor-model regression conventions standard in the literature. This project was completed as the final assignment for PGI Quant Education (Spring 2026), taught by Forrest Gao.
