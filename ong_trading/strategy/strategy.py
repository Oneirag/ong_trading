"""
Based on https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-IV/
"""
# strategy.py

import numpy as np

from abc import ABC, abstractmethod

import pandas as pd

from ong_trading.event_driven.event import SignalEvent, MarketEvent
from ong_trading.event_driven.utils import DirectionType
from ong_trading import logger


class Strategy(ABC):
    """
    Strategy is an abstract base class providing an interface for
    all subsequent (inherited) strategy handling objects.

    The goal of a (derived) Strategy object is to generate Signal
    objects for particular symbols based on the inputs of Bars
    (OLHCVI) generated by a DataHandler object.

    This is designed to work both with historic and live data as
    the Strategy object is agnostic to the data source,
    since it obtains the bar tuples from a queue object.
    """

    logger = logger

    def __init__(self, bars, events):
        self.bars = bars
        self.symbol_list = self.bars.symbol_list
        self.events = events

    @abstractmethod
    def calculate_signals(self, event):
        """
        Provides the mechanisms to calculate the list of signals.
        """
        raise NotImplementedError("Should implement calculate_signals()")


class BuyAndHoldStrategy(Strategy):
    """
    This is an extremely simple strategy that goes LONG all the
    symbols as soon as a bar is received. It will never exit a position.

    It is primarily used as a testing mechanism for the Strategy class
    as well as a benchmark upon which to compare other strategies.
    """

    def __init__(self, bars, events):
        """
        Initialises the buy and hold strategy.

        Parameters:
        bars - The DataHandler object that provides bar information
        events - The Event Queue object.
        """
        super(Strategy).__init__(bars, events)

        # Once buy & hold signal is given, these are set to True
        self.bought = self._calculate_initial_bought()

    def _calculate_initial_bought(self):
        """
        Adds keys to the bought dictionary for all symbols
        and sets them to False.
        """
        bought = {}
        for s in self.symbol_list:
            bought[s] = False
        return bought

    def calculate_signals(self, event):
        """
        For "Buy and Hold" we generate a single signal per symbol
        and then no additional signals. This means we are
        constantly long the market from the date of strategy
        initialisation.

        Parameters
        event - A MarketEvent object.
        """
        #if event.type == 'MARKET':
        if isinstance(event, MarketEvent):
            for s in self.symbol_list:
                bars = self.bars.get_latest_bars(s, N=1)
                if bars is not None and bars != []:
                    if not self.bought[s]:
                        # (Symbol, Datetime, Type = LONG, SHORT or EXIT)
                        # signal = SignalEvent(bars[0][0], bars[0][1], 'LONG')
                        signal = SignalEvent(bars[0].symbol, bars[0].timestamp, DirectionType.BUY)
                        self.events.put(signal)
                        # self.bought[s] = True     # Comment this part to send permanently a buy signal


class MACrossOverStrategy(Strategy):
    """
    Simple crossover of short and long moving averages
    """
    def __init__(self, bars, events, short=5, long=20):
        super().__init__(bars, events)
        self.short = short
        self.long = long
        self.positions = {s: 0 for s in self.bars.symbol_list}
        self.last_signal = 0

    def calculate_signals(self, event):
        # Calculate both means
        for symbol in self.bars.symbol_list:
            long_bars = self.bars.get_latest_bars(symbol, N=self.long)
            short_bars = self.bars.get_latest_bars(symbol, N=self.short)
            if len(long_bars) == self.long:
                short = np.mean(list(bar.close for bar in short_bars))
                long = np.mean(list(bar.close for bar in long_bars))
                if short >= long:
                    # self.logger.debug(f"Go long in {symbol} in date {short_bars[-1].timestamp}")
                    signal = SignalEvent(symbol, short_bars[-1].timestamp, DirectionType.BUY)
                else:
                    # self.logger.debug(f"Go short in {symbol} in date {short_bars[-1].timestamp}")
                    signal = SignalEvent(symbol, short_bars[-1].timestamp, DirectionType.SELL)
                # Avoid sending multiple signals in the same sense
                if signal.signal_type.value != self.last_signal:
                    self.last_signal = signal.signal_type.value
                    self.events.put(signal)


class PersistenceStrategy(Strategy):
    """
    simple persistence strategy: if yesterday stock price rose, then buy, if decreased then sell and do
    nothing if change is below a certain % threshold of last price
    """

    def __init__(self, bars, events, threshold_pct: float = 0):
        super().__init__(bars, events)
        self.threshold_pct = threshold_pct

    def calculate_signals(self, event):
        for symbol in self.bars.symbol_list:
            last_2_bar = self.bars.get_latest_bars(symbol, N=2)
            if len(last_2_bar) >= 2:
                last_price = last_2_bar[-1].close
                change = last_price - last_2_bar[-2].close
                timestamp = last_2_bar[-1].timestamp
                if abs(change) < last_price * self.threshold_pct:
                    direction = DirectionType.NEUTRAL
                elif change > 0:
                    direction = DirectionType.BUY
                else:
                    direction = DirectionType.SELL
                signal = SignalEvent(symbol, timestamp, direction)
                self.events.put(signal)


class ManualStrategy(Strategy):
    """Class for a strategy that is fed using a pandas dataframe instead of being calculated"""

    def __init__(self, bars, events, strategy_df: pd.DataFrame):
        super(Strategy).__init__(bars, events)
        self.df_strategy = strategy_df
        if self.df_strategy.columns.tolist() != list(self.symbol_list):
            raise ValueError("Dataframe columns do not match all symbols")

    def calculate_signals(self, event):
        for symbol in self.symbol_list:
            signal_value = self.df_strategy.at[event.timestamp, symbol]
            if signal_value > 0:
                direction = DirectionType.BUY
            elif signal_value < 0:
                direction = DirectionType.SELL
            else:
                direction = DirectionType.NEUTRAL
            signal = SignalEvent(symbol, event.timestamp, direction)
            self.events.put(signal)
