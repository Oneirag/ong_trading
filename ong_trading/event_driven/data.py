"""
Based on https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-III/
"""
# data.py
from __future__ import annotations

import datetime
import os
import os.path
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
import yfinance as yf

from ong_trading import logger
from ong_trading.event_driven.event import MarketEvent, BacktestingEndEvent
from ong_trading.event_driven.rates import Rates

prepare_data_cfg = {
    "ELE.MC": {
        "start_date": "2001-01-01",  # remove data prior to that date
        # "dividend_threshold": 0.15  # dividend threshold
        "dividend_threshold": 0.000001  # dividend threshold
    }
}


def prepare_data(df: pd.DataFrame, start_date, dividend_threshold):
    # remove year 2000
    df = df[start_date:]
    idx_dividends = df.Dividends > 0
    # look for huge dividends (extraordinaries)
    divs = df[idx_dividends]
    huge_divs = divs[(divs.Dividends / df.Close.shift(1)[idx_dividends]) > dividend_threshold]
    for idx, row in huge_divs.iterrows():
        factor = 1 - divs.Dividends[idx] / df.Close.shift(1)[idx]
        df.loc[:idx - pd.offsets.BDay(1), ['Open', 'High', 'Low', "Close"]] *= factor
        # df.loc[:idx - pd.offsets.BDay(1), ['Open', 'High', 'Low', "Close"]] -= row.Dividends  # Not return - adjusted
        df.loc[idx, "Dividends"] = 0  # Remove dividend from current list
    return df


@dataclass
class Bar:
    symbol: str
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: float = 0
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

    def __init__(self):
        self.continue_backtest = True  # For event_driven to work in any child class

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
    names = ['datetime', 'open', 'low', 'high', 'close', 'volume', 'oi',
             'dividends', 'stock splits', 'adj_close']

    def __init__(self, events, csv_dir, symbol_list, bid_offer: dict = None, start_date: pd.Timestamp = None,
                 end_date: pd.Timestamp = None):
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
        start_date: date from with start reading data
        end_date: date up to with start reading data
        """
        super().__init__()

        self.events = events
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        default_start_date = pd.Timestamp(1900, 1, 1)
        self.start_date = start_date or default_start_date
        self.end_date = end_date or pd.Timestamp.now()

        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.bid_offer = bid_offer or {s: 0 for s in self.symbol_list}
        self._open_convert_csv_files()
        self.bar_data_iterator = dict()
        self.bar_data = {s: tuple(b for b in self._get_new_bar(s)) for s in self.symbol_list}
        # update start date if it was None
        if start_date is None:
            self.start_date = self.bar_data[self.symbol_list[0]][0].timestamp
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
            new_data = self._read_symbol_data(s)
            if new_data.empty:
                raise ValueError(f'Empty data for symbol "{s}". Review symbol name, dates, internet connection, etc')
            # Remove TZ information
            new_data.index = new_data.index.tz_localize(None)
            self.symbol_data[s] = new_data
            # Reads always all data no mater of start_date
            # if self.start_date:
            #     # Data does not start the first day but 150 trading days before
            #     self.symbol_data[s] = self.symbol_data[s][(self.start_date - pd.offsets.BDay(150)):]

            # Combine the index to pad forward values
            if comb_index is None:
                comb_index = self.symbol_data[s].index
            else:
                comb_index.union(self.symbol_data[s].index)

            # Set the latest symbol_data to None
            self.latest_symbol_data[s] = []

        # Reindex the dataframes
        for s in self.symbol_list:
            if self.symbol_data[s].index.tz != comb_index.tz:
                self.symbol_data[s].index = self.symbol_data[s].index.tz_localize(comb_index.tz)
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
            stock_splits = b[1].get("stock_splits", 0)
            adj_close = b[1].get("adj_close", 0)
            # bid_offer = getattr(self, "bid_offer", dict()).get(symbol, 0)
            bid_offer = self.bid_offer.get(symbol, 0)
            bar = Bar(symbol=symbol, timestamp=timestamp, open=open, high=high, low=low, close=close, volume=volume,
                      dividends=dividends, splits=stock_splits, adj_close=adj_close,
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
                while True:
                    bar = next(self.bar_data_iterator[s])
                    if bar.timestamp > self.end_date:
                        raise StopIteration()
                    if bar.timestamp < self.start_date:
                        self.latest_symbol_data[s].append(bar)
                    else:
                        break
            except StopIteration:
                self.continue_backtest = False
                self.events.put(BacktestingEndEvent())
            else:
                if bar is not None:
                    self.latest_symbol_data[s].append(bar)
                    self.events.put(MarketEvent(timestamp=bar.timestamp))
                    # self.events.put(MarketEvent(timestamp=bar[1]))

    def reset(self):
        """Reset the state of bars so a new event_driven can start without reading again all data"""
        self.continue_backtest = True
        self.latest_symbol_data = {s: list() for s in self.symbol_list}

        for s in self.symbol_list:
            self.bar_data_iterator[s] = iter(self.bar_data[s])

    @property
    def n_bars(self) -> int:
        """Gets the total number of bars (needed for splitting data for backtesting)"""
        return len(self.bar_data[self.symbol_list[0]])

    def to_pandas(self, symbol) -> pd.DataFrame:
        """Gets all historical data of a certain symbol as a pandas DataFrame indexed by timestamp"""
        df = pd.DataFrame(pd.DataFrame(self.bar_data[symbol]))
        df.set_index("timestamp", inplace=True)
        return df


class YahooHistoricalData(HistoricCSVDataHandler):
    """Class to read data directly from Yahoo using yfinance"""

    def __init__(self, events, symbol_list: list | str, period="max", bid_offer: dict = None,
                 start_date: pd.Timestamp = None, end_date: pd.Timestamp = None):
        """
        Initializes symbols
        :param events: a queue for sending events
        :param symbol_list: list of symbols to download
        :param period: period to pass to method history of a yahoo finance Ticker object (defaults to "max")
        :param bid_offer: A dictionary indexed by symbol with half of the bid_offer (the number to add/substract from
        mid to calculate ask or bid respectively). By default, 0
        :param start_date: Date from which start reading data
        :param end_date: Date up to which start reading data
        """
        if isinstance(symbol_list, str):
            symbol_list = [symbol_list]
        self.period = period
        if bid_offer is None:
            bid_offer = dict()
        bid_offer = {s: bid_offer.get(s, 0) for s in symbol_list}
        super().__init__(events, None, symbol_list, bid_offer=bid_offer, start_date=start_date, end_date=end_date)
        # Download interest rates
        bars = list(self.bar_data.values())[0]
        bars_idx = [b.timestamp for b in bars]
        self.rates = None
        self.rates = Rates(index=bars_idx)

    def _read_symbol_data(self, symbol, auto_adjust=False):
        """
        Read symbol data
        :param symbol: Name of the ticker to read
        :param auto_adjust: adjust prices according to splits and dividends. Defaults to *False* so not adjusted
        :return: a pandas Dataframe with all columns in lower letters
        """
        # Write a cache file just in case
        pkl_file = os.path.join(os.path.dirname(__file__), "cache", symbol + ".pkl")
        # pkl_file = os.path.join(tempfile.gettempdir(), symbol + ".pkl")
        if os.path.isfile(pkl_file):
            cached_data = pd.read_pickle(pkl_file)
            last_date = cached_data.index[-1]
        else:
            cached_data = None

        # if no data has been downloaded or there is a gap in downloaded data, download again

        if cached_data is None or (
                ((pd.Timestamp.today(last_date.tz).normalize() - pd.offsets.BDay(1)) - last_date).delta > 0):
            # Download data again
            dt = yf.Ticker(symbol)
            data = dt.history(period=self.period, auto_adjust=auto_adjust)
            # merge cached data with the old data
            if cached_data is not None:
                new_data = data[data.index.get_loc(cached_data.index[-1]) + 1:]
                data = pd.concat([cached_data, new_data])
            if symbol in prepare_data_cfg:
                data = prepare_data(data, **prepare_data_cfg[symbol])
        else:
            data = cached_data

        os.makedirs(os.path.dirname(pkl_file), exist_ok=True)
        data.to_pickle(pkl_file)
        # change order to match self.names
        data.columns = [c.lower().replace(" ", "_") for c in data.columns]
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
