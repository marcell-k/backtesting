from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from backtesting import Backtest


def calculate_risk_reward_ratio(trades_df: pd.DataFrame) -> pd.DataFrame:
    df = trades_df.copy()
    is_long = df["Size"] > 0
    size_abs = df["Size"].abs()

    df["PotentialRisk"] = np.where(
        is_long, (df["EntryPrice"] - df["OpenSL"]) * size_abs, (df["OpenSL"] - df["EntryPrice"]) * size_abs
    )

    df["RealizedReward"] = np.where(
        is_long,
        size_abs * (df["ExitPrice"] - df["EntryPrice"]) - df["Commission"],
        size_abs * (df["EntryPrice"] - df["ExitPrice"]) - df["Commission"],
    )

    df["RiskRewardRatio"] = df["RealizedReward"] / df["PotentialRisk"].replace(0, np.nan)
    df["RiskRewardRatio"] = pd.to_numeric(df["RiskRewardRatio"], errors="coerce")
    df["RiskRewardRatio"] = df["RiskRewardRatio"].replace([np.inf, -np.inf], np.nan)
    return df


def _auto_scale(*values: float) -> tuple[float, str]:
    """Pick a K/M/B divisor based on the largest magnitude among `values`."""
    largest = max(abs(v) for v in values)
    if largest >= 1e9:
        return 1e9, "B"
    if largest >= 1e6:
        return 1e6, "M"
    if largest >= 1e3:
        return 1e3, "K"
    return 1, ""


def get_ambiguous_fill_trades(bt: Backtest | None = None, **run_kwargs) -> pd.DataFrame:
    """Return the subset of trades affected by the "ambiguous same-bar SL/TP" warning."""
    import warnings

    if bt is None:
        raise ValueError("Must pass `bt`")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        stats = bt.run(**run_kwargs)

    ambiguous_times = {
        str(warning.message).split("(")[1].split(")")[0] for warning in w if "contingent SL/TP" in str(warning.message)
    }

    trades = stats["_trades"]
    trades = calculate_risk_reward_ratio(trades)
    affected = trades[trades["EntryTime"].astype(str).isin(ambiguous_times)]

    title = "Ambiguous Fill Trades"
    col_header = f"{'EntryTime':<28} | {'RRR':>6}"
    width = max(len(title), len(col_header))

    print(title)
    print("-" * width)
    print(col_header)
    print("-" * width)
    for _, row in affected.iterrows():
        print(f"{row['EntryTime']!s:<28} | {row['RiskRewardRatio']:>6.2f}")
    print("-" * width)
    print()

    n_affected, n_total = len(affected), len(trades)
    print(f"Affected:      {n_affected:>6} / {n_total:<6} trades  ({100 * n_affected / n_total:.2f}%)")

    pnl_affected, pnl_total = affected["PnL"].sum(), trades["PnL"].sum()
    pnl_pct = 100 * pnl_affected / pnl_total if pnl_total else float("nan")
    div, suffix = _auto_scale(pnl_affected, pnl_total)
    print(f"PnL:      {pnl_affected / div:>9,.2f}{suffix} / {pnl_total / div:<9,.2f}{suffix}  ({pnl_pct:.2f}%)")

    vol_affected = affected["PnL"].abs().sum()
    vol_total = trades["PnL"].abs().sum()
    vol_pct = 100 * vol_affected / vol_total if vol_total else 0.0
    div, suffix = _auto_scale(vol_affected, vol_total)
    print(f"Volume (abs): {vol_affected / div:>9,.2f}{suffix} / {vol_total / div:<9,.2f}{suffix}  ({vol_pct:.2f}%)")

    return affected
