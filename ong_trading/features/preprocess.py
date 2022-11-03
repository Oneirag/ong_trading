"""
Classes for preprocessing data
"""
from __future__ import annotations

import os.path

import pandas as pd
import talib
from ong_trading import logger
import abc
from ong_trading.event_driven.data import DataHandler, HistoricCSVDataHandler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import plotly
import plotly.graph_objects as go
import pickle


class MLPreprocessor:
    """Base class for preprocessing data"""

    # These are the columns that all input dataframe should have
    columns_of_interest = ['adj_close', 'close', 'open', 'high', 'low', 'volume']
    window_size = 200        # Size of the window for preprocessing (number of past days to include in preprocessing)

    def preprocess_bars(self, bars: DataHandler, symbol: str):
        bars = bars.get_latest_bars(symbol, N=self.window_size)  # Return last window_size bars
        if len(bars) < self.window_size:
            return None
        else:
            df = pd.DataFrame(bars)
            df = df.set_index("timestamp").loc[:, ['adj_close', 'close', 'volume', 'low', 'high']]
            proc = self.preprocess_df(df)
            if proc is None:
                return None
            last_row = proc.iloc[-1, :]
            # Assert that last timestamp of bars and proc are the same
            # assert bars[-1].timestamp == proc.index[-1]
            return last_row.values.reshape((1, last_row.shape[-1]))
        pass

    @abc.abstractmethod
    def preprocess_df(self, data: pd.DataFrame) -> pd.DataFrame:
        """To be implemented in children classes"""
        pass

    @classmethod
    def __get_filename(cls, path: str) -> str:
        filename = os.path.join(path, cls.__name__ + ".pkl")
        return filename

    def save(self, path: str):
        """Save preprocessor in the given path"""
        filename = self.__get_filename(path)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str):
        """Loads preprocessor from the given path"""
        filename = cls.__get_filename(path)
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        cls_classname = cls.__name__
        obj_classname = obj.__class__.__name__
        if obj_classname != cls_classname:
            raise ValueError(f"Error loading preprocessor. Expected class {cls_classname} but loaded {obj_classname}")
        return obj

    def test(self, data: HistoricCSVDataHandler) -> dict:
        """
        Tests the transformation: fits all data, plots it
        :param data:
        :return: a dict indexed by symbol of all transformed data
        """
        retval = dict()
        for symbol in data.symbol_list:
            in_df = data.to_pandas(symbol)
            processed = self.preprocess_df(in_df)
            in_df = in_df.loc[:, ["open", "high", "low", "close"]]
            retval[symbol] = processed
            from plotly.subplots import make_subplots
            fig = plotly.subplots.make_subplots(rows=2, cols=1, shared_xaxes=True,
                                                subplot_titles=[f"{symbol} - Original", f"{symbol} - Transformed"])

            for idx, df in enumerate((in_df, processed)):
                for column in df.columns:
                    fig.add_trace(go.Scatter(x=df.index, y=df[column], name=column), row=idx+1, col=1)
            fig.show()
        return retval


class RLPreprocessorClose(MLPreprocessor):
    """
    Preprocessor for Reinforcement Learning from github book
    https://github.com/Oneirag/Machine-Learning-for-Algorithmic-Trading-Second-Edition/blob/master/22_deep_reinforcement_learning/04_q_learning_for_trading.ipynb
    """

    def __init__(self, data: HistoricCSVDataHandler | None, validation_window_len=252,
                 normalize=True, print_log=False, window_size=None, close_column="close"):
        self.normalize = normalize
        self.print_log = print_log
        self.window_size = window_size or self.window_size
        self.close_column = close_column
        self.scaler = None
        if data is not None:
            # Validate all data to fit scaler
            df = data.to_pandas(data.symbol_list[0]).iloc[:-validation_window_len]
            self.preprocess_df(df)

    def preprocess_df(self, data: pd.DataFrame) -> pd.DataFrame | None:
        if data.shape[1] > len(self.columns_of_interest):
             data = data.loc[:, self.columns_of_interest]
        close = data.loc[:, self.close_column]

        retval = pd.DataFrame(close.pct_change()).rename(columns=dict(close="returns"))
        retval['ret_2'] = close.pct_change(2)
        retval['ret_5'] = close.pct_change(5)
        retval['ret_10'] = close.pct_change(10)
        retval['ret_21'] = close.pct_change(21)
        retval['rsi'] = talib.STOCHRSI(close)[1]
        retval['macd'] = talib.MACD(close)[1]

        # These indicators are constructed with the original close...
        retval['atr'] = talib.ATR(data.high, data.low, data.close)
        slowk, slowd = talib.STOCH(data.high, data.low, data.close)
        retval['stoch'] = slowd - slowk
        retval['ultosc'] = talib.ULTOSC(data.high, data.low, data.close)
        retval = retval.replace((np.inf, -np.inf), np.nan).dropna()
        if retval.empty:
            return None
        if self.normalize:
            scale_values = retval.iloc[:, 1:].values
            if self.scaler is None:
                self.scaler = StandardScaler()
                self.scaler.fit(scale_values)
            retval.iloc[:, 1:] = self.scaler.transform(scale_values)
        if self.print_log:
            logger.info(retval.info())
        return retval


class RLPreprocessorAdjClose(RLPreprocessorClose):
    """
    Same as parent but using adjusted close column instead of close
    """

    def __init__(self, normalize=True, print_log=False, window_size=None):
        super(RLPreprocessorAdjClose, self).__init__(normalize, print_log, window_size, close_column="adj_close")


class PCA_Preprocessor(MLPreprocessor):
    """
    Uses principal component analysis to reduce number of data. It is meant just for one symbol!!!
    """

    # column_names = ['adj_close', 'open', 'high', 'low']
    column_names = ['close', 'open', 'high', 'low']

    def __init__(self, data: HistoricCSVDataHandler, n_pca_components=10,
                 validation_window_len=252, window_size=30, symbol_name: str = None):
        """
        Fits a pca to the first OHLC data of a certain window prior to current data of given symbol
        :param data: a HistoricCSVDataHandler with the data of interest
        :param n_pca_components: number of PCA components to retain
        :param validation_window_len: number of days of validation data (so PCA won't be fitted using these data)
        :param window_size: number of days of the window that will be reduced using PCA (defaults to 30)
        """
        self.n_pca_components = n_pca_components
        self.window_size = window_size
        self.validation_window_len = validation_window_len
        self.symbol = symbol_name or data.symbol_list[0]
        df = data.to_pandas(self.symbol)
        # Do not use last year to scale
        df = df.iloc[:-validation_window_len]
        x = self.__prepare_df_for_pca(df)
        # self.pre_scaler = StandardScaler()
        self.pre_scaler = MinMaxScaler()
        self.pre_scaler.fit(x)
        X = self.pre_scaler.transform(x)
        self.pca = PCA(n_components=self.n_pca_components)
        self.pca.fit(X)
        pca_out = self.pca.transform(X)
        # self.post_scaler = StandardScaler()
        self.post_scaler = MinMaxScaler()
        self.post_scaler.fit(pca_out)

    def __prepare_df_for_pca(self, df_pca: pd.DataFrame) -> np.array:
        """Transforms input dataframe into a matrix that can be used for pca analysis """
        in_df = df_pca.loc[:, self.column_names]
        n_cols = in_df.shape[1]
        in_pca = in_df.values.flatten()
        n_repeats = df_pca.shape[0] - self.window_size
        mask = np.tile(np.arange(self.window_size * n_cols)[None, :], (n_repeats, 1)) + \
               n_cols * np.arange(n_repeats)[:, None]
        x = in_pca[mask]
        return x

    def test(self, data: HistoricCSVDataHandler) -> dict:
        processed_dict = super(PCA_Preprocessor, self).test(data)
        return processed_dict
        for symbol, processed in processed_dict.items():
            X = processed.values
            # Reconstruct variable
            pca_X = self.pca.transform(X)
            # Create two subplots and unpack the output array immediately
            plot_len = 100
            f, ax = plt.subplots(2, 2, sharex="all")
            for idx_axis, (what_to_plot, title) in enumerate((
                    (X, 'Original (no reconstructed)'),
                    (pca_X, 'PCA (no reconstructed)'),
                    (self.scaler.inverse_transform(X), 'Original (reconstructed)'),
                    (self.scaler.inverse_transform(pca.inverse_transform(pca_X)), 'PCA (reconstructed)')
            )):
                axis = np.array(ax).flatten()[idx_axis]
                axis.set_title(title)
                axis.plot(what_to_plot[:plot_len])
            plt.show()

            plt.bar(x=range(n_components + 1), height=pca.singular_values_[:])
            plt.title("Singular Values")
            plt.show()
            plt.bar(x=range(n_components + 1), height=pca.explained_variance_ratio_[:].T)
            plt.title("Explained values")
            plt.show()
            plt.plot(pca.components_[:].T, label=np.arange(n_components + 1))
            plt.title("Components")
            plt.legend()
            plt.show()
        return processed_dict

    def preprocess_df(self, data: pd.DataFrame) -> pd.DataFrame:
        df_in = data.loc[:, self.column_names]
        x = self.__prepare_df_for_pca(df_in)
        X = self.pre_scaler.transform(x)
        pca = self.pca.transform(X)
        out_pca = self.post_scaler.transform(pca)
        return pd.DataFrame(out_pca, index=data.index[self.window_size:])


if __name__ == '__main__':
    from ong_trading.event_driven.data import YahooHistoricalData
    data = YahooHistoricalData(None, "ELE.MC")

    rl = RLPreprocessorClose()
    rl.test(data)

    rladj = RLPreprocessorAdjClose()
    rladj.test(data)

    pca = PCA_Preprocessor(data)
    pca.test(data)


