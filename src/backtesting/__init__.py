from ._util import try_
from .backtesting import Backtest, Strategy

__all__ = ["Backtest", "Pool", "Strategy"]


def Pool(processes=None, initializer=None, initargs=()):  # noqa
    import multiprocessing as mp
    import sys

    # Revert performance related change in Python>=3.14
    if sys.platform.startswith("linux") and mp.get_start_method(allow_none=True) != "fork":
        try_(lambda: mp.set_start_method("fork"))
    if mp.get_start_method() == "spawn":
        import warnings

        warnings.warn(
            "If you want to use multi-process optimization with "
            "`multiprocessing.get_start_method() == 'spawn'` (e.g. on Windows),"
            "set `backtesting.Pool = multiprocessing.Pool` (or of the desired context) "
            "and hide `bt.optimize()` call behind a `if __name__ == '__main__'` guard. "
            "Currently using thread-based paralellism, "
            "which might be slightly slower for non-numpy / non-GIL-releasing code. "
            "See https://github.com/kernc/backtesting.py/issues/1256",
            category=RuntimeWarning,
            stacklevel=3,
        )
        from multiprocessing.dummy import Pool

        return Pool(processes, initializer, initargs)
    else:
        return mp.Pool(processes, initializer, initargs)
