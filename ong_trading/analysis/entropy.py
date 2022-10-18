"""
Calculates the entropy of some indicators
"""
import pandas as pd

from ong_trading.event_driven.backtester import SignalAnalysisBacktester

from ong_trading.event_driven.data import YahooHistoricalData
from ong_trading.strategy.strategy import MACrossOverStrategy, PersistenceStrategy
import numpy as np
from itertools import product


def compute_entropy(df: pd.DataFrame, nbins: int):
    """Computes entropy of an indicator, for the given number of bins"""
    v = df.values
    n = len(df.values)
    bins, edges = np.histogram(v, nbins)
    freqs = bins / n
    entropy = -sum(p * np.log(p) if p else 0 for p in freqs) / np.log(nbins)
    return entropy


if __name__ == '__main__':

    params = [
        # (MACrossOverStrategy, product(range(3, 10), range(7, 30))),
        (PersistenceStrategy, ((0,), (0.01,), (0.005,), (0.0005,)))
    ]

    for strategy_class, strategy_args_list in params:
        print("*" * 50)
        print(strategy_class.__name__)
        print("*" * 50)
        bt = SignalAnalysisBacktester(
            data_class=YahooHistoricalData,
            strategy_class=strategy_class,
            symbol_list="ELE.MC",
            start_date="2020-01-01"
        )

        dict_entropy = dict()
        for strategy_args in strategy_args_list:
            res = bt.run(strategy_args=strategy_args)
            signals = bt.get_signals()
            # print(signals)
            entropy = compute_entropy(signals, nbins=3)
            dict_entropy[strategy_args] = entropy
            print(strategy_args, entropy)

