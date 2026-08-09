import time
import tracemalloc

import numpy as np
import pandas as pd

from backtesting import Backtest, Strategy
from backtesting.lib import crossover

rng = np.random.default_rng(42)
n = 1_000_000

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


def SMA(values, n) -> pd.Series:
    return pd.Series(values).rolling(n).mean()


class SmaCross(Strategy):
    n1 = 10
    n2 = 20

    def init(self) -> None:
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)

    def next(self) -> None:
        if crossover(self.sma1, self.sma2):
            self.position.close()
            self.buy(size=0.0001)

        elif crossover(self.sma2, self.sma1):
            self.position.close()
            self.sell(size=0.0001)


bt = Backtest(df, SmaCross, margin=1 / 100, cash=10_000_000, commission=0.0000001, finalize_trades=True)

stats = bt.run()
print(stats)

WARMUP = 0
N = 1

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


# Original Benchmark results
# -----------------
# Warmup runs: 5
# Measured runs: 70
#
# Execution time (seconds)
# Mean:   4.904529
# Median: 4.854094
# 95%:    5.149979
#
# Peak Python memory allocation (MB)
# Mean:   38.26
# Median: 38.26
# 95%:    38.27

# Return [%]                          -17.10126
# Buy & Hold Return [%]                -1.93959
# Return (Ann.) [%]                    -0.02363
# Volatility (Ann.) [%]                 0.12925
# CAGR [%]                             -0.01688
# Sharpe Ratio                         -0.18282
#


# New
# Benchmark results
# -----------------
# Warmup runs: 5
# Measured runs: 20
#
# Execution time (seconds)
# Mean:   4.167699
# Median: 4.154688
# 95%:    4.209478
#
# Peak Python memory allocation (MB)
# Mean:   38.25
# Median: 38.25
# 95%:    38.26


# Warmup runs: 5
# Measured runs: 70
#
# Execution time (seconds)
# Mean:   3.621650
# Median: 3.602665
# 95%:    3.667898
#
# Peak Python memory allocation (MB)
# Mean:   38.25
# Median: 38.25
# 95%:    38.26
