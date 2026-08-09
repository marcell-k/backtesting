import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from backtesting import Backtest, Strategy
from backtesting.lib import crossover

DATA_PATH = Path("data.csv")
if DATA_PATH.exists():
    df = pd.read_csv("data.csv", index_col="Date", parse_dates=True)
else:
    rng = np.random.default_rng(42)
    n = 300_000

    dates = pd.bdate_range("2000-01-03", periods=n)
    target_price = 100.0  # Long-term mean price
    theta = 0.005  # Speed of mean reversion
    sigma = 0.8  # Daily price volatility
    dt = 1.0
    decay = np.exp(-theta * dt)
    noise_std = sigma * np.sqrt((1 - np.exp(-2 * theta * dt)) / (2 * theta))
    shocks = rng.normal(0, noise_std, n)
    shocks[0] = 0

    close = np.zeros(n)
    close[0] = target_price
    for i in range(1, n):
        close[i] = close[i - 1] * decay + target_price * (1 - decay) + shocks[i]
    close = np.maximum(close, 1.0)
    open_ = close * (1 + rng.normal(0, 0.002, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n)))

    df = pd.DataFrame(
        {
            "Open": open_.astype(np.float32),
            "High": high.astype(np.float32),
            "Low": low.astype(np.float32),
            "Close": close.astype(np.float32),
            "Volume": rng.integers(1_000_000, 50_000_000, n, dtype=np.int32),
        },
        index=dates,
    )
    df.index.name = "Date"
    df.to_csv("data.csv")

def SMA(values:pd.Series, n:int) -> pd.Series:
    return pd.Series(values).rolling(n).mean()


class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self) -> None:
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)

    def next(self) -> None:
        if crossover(self.sma1, self.sma2): # pyright: ignore[]
            self.position.close()
            self.buy(size=0.0001)

        elif crossover(self.sma2, self.sma1): # pyright: ignore[]
            self.position.close()
            self.sell(size=0.0001)


bt = Backtest(df, SmaCross, margin=1 / 100, cash=10_000_000, commission=0.0000001, finalize_trades=True)

stats = bt.run()
# print(stats)
WARMUP = 3
N = 10

for _ in range(WARMUP):
    bt.run()

times = []
memory_peaks = []

for _ in range(N):
    tracemalloc.start()

    start = time.perf_counter()

    bt.run()

    elapsed = time.perf_counter() - start

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    times.append(elapsed)
    memory_peaks.append(peak / 1024 / 1024)  # MB


times = np.array(times)
memory_peaks = np.array(memory_peaks)

print("\nBenchmark results")
print("-----------------")
print(f"Warmup runs: {WARMUP}")
print(f"Measured runs: {N}")

print("\nExecution time (seconds)")
print(f"Mean:   {times.mean():.6f}")
print(f"Median: {np.median(times):.6f}")
print(f"95%:    {np.percentile(times, 95):.6f}")

print("\nPeak Python memory allocation (MB)")
print(f"Mean:   {memory_peaks.mean():.2f}")
print(f"Median: {np.median(memory_peaks):.2f}")
print(f"95%:    {np.percentile(memory_peaks, 95):.2f}")


with Path("results.txt").open("w") as f:
    f.write("Backtest stats\n--------------\n")
    f.write(repr(stats))
    f.write("\n\n\n\n\n\n")

    f.write("Benchmark results\n-----------------\n")
    f.write(f"Warmup runs: {WARMUP}\n")
    f.write(f"Measured runs: {N}\n\n")
    f.write("Execution time (seconds)\n")
    f.write(f"Mean:   {times.mean():.6f}\n")
    f.write(f"Median: {np.median(times):.6f}\n")
    f.write(f"95%:    {np.percentile(times, 95):.6f}\n\n")
    f.write("Peak Python memory allocation (MB)\n")
    f.write(f"Mean:   {memory_peaks.mean():.2f}\n")
    f.write(f"Median: {np.median(memory_peaks):.2f}\n")
    f.write(f"95%:    {np.percentile(memory_peaks, 95):.2f}\n")

# Warmup runs: 3
# Measured runs: 20
#Execution time (seconds)
# Mean:   4.929233
# Median: 4.860259
# 95%:    5.344519
#
# Peak Python memory allocation (MB)
# Mean:   57.38
# Median: 57.38
# 95%:    57.38

# Start                     2000-01-03 00:00:00
# End                       3149-12-02 00:00:00
# Duration                  419997 days 00:0...
# Exposure Time [%]                    99.97967
# Equity Final [$]                7527925.05895
# Equity Peak [$]                10009035.05891
# Commissions [$]                     294.52961
# Return [%]                          -24.72075
# Buy & Hold Return [%]                 6.33284
# Return (Ann.) [%]                    -0.02385
# Volatility (Ann.) [%]                 0.12884
# CAGR [%]                             -0.01704
# Sharpe Ratio                         -0.18512
# Sortino Ratio                        -0.25925
# Calmar Ratio                         -0.00095
# Alpha [%]                           -24.72034
# Beta                                 -0.00006
# Max. Drawdown [%]                   -25.09911
# Avg. Drawdown [%]                    -2.79675
# Max. Drawdown Duration    419836 days 00:0...
# Avg. Drawdown Duration    46656 days 00:00:00
# # Trades                                16982
# Win Rate [%]                         38.81757
# Best Trade [%]                        26.4339
# Worst Trade [%]                      -9.52865
# Avg. Trade [%]                       -0.21595
# Max. Trade Duration         186 days 00:00:00
# Avg. Trade Duration          25 days 00:00:00
# Profit Factor                         0.86491
# Expectancy [%]                        -0.1668
# SQN                                  -6.84576
# Kelly Criterion                      -0.06101
