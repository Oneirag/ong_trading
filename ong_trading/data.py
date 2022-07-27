"""
Based on https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-III/
"""
# data.py

import datetime
import os
import os.path
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from ong_trading import logger
from ong_trading.event import MarketEvent
from ong_trading.rates import Rates


@dataclass
class Bar:
    symbol: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    dividends: float = 0
    splits: float = 0
    bid_ask_open: float = 0
    bid_ask_high: float = 0
    bid_ask_low: float = 0
    bid_ask_close: float = 0

    def __post_init__(self):
        for name in "open", "high", "low", "close":
            for bid_ask, sign_bid_ask in {"bid": -1, "ask": 1}.items():
                price_value = getattr(self, name)
                spread_value = getattr(self, f"bid_ask_{name}")
                value = price_value + sign_bid_ask * spread_value
                setattr(self, f"{name}_{bid_ask}", value)


class DataHandler(ABC):
    """
    DataHandler is an abstract base class providing an interface for
    all subsequent (inherited) data handlers (both live and historic).

    The goal of a (derived) DataHandler object is to output a generated
    set of bars (OLHCVI) for each symbol requested.

    This will replicate how a live strategy would function as current
    market data would be sent "down the pipe". Thus a historic and live
    system will be treated identically by the rest of the backtesting suite.
    """
    logger = logger

    @abstractmethod
    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or fewer if less bars are available.
        """
        raise NotImplementedError("Should implement get_latest_bars()")

    @abstractmethod
    def update_bars(self):
        """
        Pushes the latest bar to the latest symbol structure
        for all symbols in the symbol list.
        """
        raise NotImplementedError("Should implement update_bars()")


class HistoricCSVDataHandler(DataHandler):
    """
    HistoricCSVDataHandler is designed to read CSV files for
    each requested symbol from disk and provide an interface
    to obtain the "latest" bar in a manner identical to a live
    trading interface.
    """

    def __init__(self, events, csv_dir, symbol_list, bid_offer: dict = None):
        """
        Initialises the historic data handler by requesting
        the location of the CSV files and a list of symbols.

        It will be assumed that all files are of the form
        'symbol.csv', where symbol is a string in the list.

        Parameters:
        events - The Event Queue.
        csv_dir - Absolute directory path to the CSV files.
        symbol_list - A list of symbol strings.
        bid_offer: a dict of bid_offer spreads indexed for each symbol
        """
        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list

        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True

        self.names = ['datetime', 'open', 'low', 'high', 'close', 'volume', 'oi',
                      'dividends', 'stock splits']
        self.bid_offer = bid_offer or {s: 0 for s in self.symbol_list}
        self._open_convert_csv_files()
        self.bar_data_iterator = dict()
        self.bar_data = {s: tuple(b for b in self._get_new_bar(s)) for s in self.symbol_list}
        self.reset()

    def _read_symbol_data(self, symbol):
        return pd.read_csv(os.path.join(self.csv_dir, f'{symbol}.csv'),
                           header=0, index_col=0,
                           names=self.names
                           )

    def _open_convert_csv_files(self):
        """
        Opens the CSV files from the data directory, converting
        them into pandas DataFrames within a symbol dictionary.

        For this handler it will be assumed that the data is
        taken from DTN IQFeed. Thus, its format will be respected.
        """
        comb_index = None
        for s in self.symbol_list:
            # Load the CSV file with no header information, indexed on date
            self.symbol_data[s] = self._read_symbol_data(s)

            # Combine the index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_data[s].index
            else:
                comb_index.union(self.symbol_data[s].index)

            # Set the latest symbol_data to None
            self.latest_symbol_data[s] = []

        # Reindex the dataframes
        for s in self.symbol_list:
            self.symbol_data[s] = self.symbol_data[s].reindex(index=comb_index, method='pad').iterrows()

    def _get_new_bar(self, symbol):
        """
        Returns the latest bar from the data feed as a tuple of
        (symbol, datetime, open, low, high, close, volume).
        """
        for b in self.symbol_data[symbol]:
            if isinstance(b[0], str):
                timestamp = datetime.datetime.strptime(b[0], '%Y-%m-%d %H:%M:%S')
            else:
                timestamp = b[0]
            open, low, high, close, volume = b[1][:5]
            dividends = b[1].get("dividends", 0)
            stock_splits = b[1].get("stock splits", 0)
            # bid_offer = getattr(self, "bid_offer", dict()).get(symbol, 0)
            bid_offer = self.bid_offer.get(symbol, 0)
            bar = Bar(symbol=symbol, timestamp=timestamp, open=open, high=high, low=low, close=close, volume=volume,
                      dividends=dividends, splits=stock_splits,
                      bid_ask_low=bid_offer, bid_ask_close=bid_offer, bid_ask_high=bid_offer, bid_ask_open=bid_offer)
            yield bar
            # yield symbol, timestamp, open, low, high, close, volume

    def get_latest_bars(self, symbol, N=1):
        """
        Returns the last N bars from the latest_symbol list,
        or N-k if less available.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            self.logger.error(f"Symbol {symbol} is not available in the historical data set.")
        else:
            if N is None:
                return bars_list
            else:
                return bars_list[-N:]

    def update_bars(self):
        """
        Pushes the latest bar to the latest_symbol_data structure
        for all symbols in the symbol list.
        """
        for s in self.symbol_list:
            try:
                # bar = next(self._get_new_bar(s))
                bar = next(self.bar_data_iterator[s])
            except StopIteration:
                self.continue_backtest = False
                self.events.put(MarketEvent())
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar)
                    self.events.put(MarketEvent(timestamp=bar.timestamp))
                    #self.events.put(MarketEvent(timestamp=bar[1]))

    def reset(self):
        """Reset the state of bars so a new backtest can start without reading again all data"""
        self.continue_backtest = True
        self.latest_symbol_data = {s: list() for s in self.symbol_list}

        for s in self.symbol_list:
            self.bar_data_iterator[s] = iter(self.bar_data[s])


class YahooHistoricalData(HistoricCSVDataHandler):
    """Class to read data directly from Yahoo using yfinance"""

    def __init__(self, events, symbol_list: list, period="max", bid_offer: dict = None):
        """
        Initializes symbols
        :param events:
        :param symbol_list:
        :param period: period to pass to method history of a yahoo finance Ticker object (defaults to "max")
        :param bid_offer: A dictionary indexed by symbol with half of the bid_offer (the number to add/substract from
        mid to calculate ask or bid respectively). By default, 0
        """
        if isinstance(symbol_list, str):
            symbol_list = [symbol_list]
        self.period = period
        if bid_offer is None:
            bid_offer = dict()
        bid_offer = {s: bid_offer.get(s, 0) for s in symbol_list}
        super().__init__(events, None, symbol_list, bid_offer=bid_offer)
        self.names.append('adj_close')      # So adjusted close is there
        # Download interest rates
        bars = list(self.bar_data.values())[0]
        bars_idx = [b.timestamp for b in bars]
        self.rates = None
        # self.rates = Rates(index=bars_idx)

    def _read_symbol_data(self, symbol, auto_adjust=False):
        """
        Read symbol data
        :param symbol: Name of the ticker to read
        :param auto_adjust: adjust prices according to splits and dividends. Defaults to *False* so not adjusted
        :return: a pandas Dataframe with all columns in lower letters
        """
        # Write a cache file just in case
        pkl_file = os.path.join(tempfile.gettempdir(), symbol + ".pkl")
        if os.path.isfile(pkl_file):
            data = pd.read_pickle(pkl_file)
        else:
            dt = yf.Ticker(symbol)
            data = dt.history(period=self.period, auto_adjust=auto_adjust)
            data.to_pickle(pkl_file)
        # change order to match self.names
        data.columns = [c.lower() for c in data.columns]
        data = data[(c for c in self.names if c in data.columns)]

        return data


class GeneratedHistoricalData(YahooHistoricalData):
    """Uses a df to generate fake historical data"""

    def __init__(self, events, symbol_list, dict_df: dict, bid_offer: dict):
        """
        Creates a fake HistoricalData source
        :param events: a Queue object
        :param symbol_list: list of symbols
        :param dict_df: a dictionary of dataframes, with symbol as key, with the OHLC data for each symbol (same columns
        as yahoo finance ones)
        :param bid_offer: a dict with the bid_offer spread
        """
        self._dict_df = dict_df
        super().__init__(events, symbol_list, bid_offer=bid_offer)

    def _read_symbol_data(self, symbol, auto_adjust=False):
        data = self._dict_df[symbol]
        data.columns = [c.lower() for c in data.columns]
        # data = data[(c for c in self.names if c in data.columns)]
        return data



