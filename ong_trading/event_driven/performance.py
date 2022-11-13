"""
Based on https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-VII/
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
from ong_trading.event_driven.utils import plot_chart, plot_maxdd
from ong_trading.utils.utils import fmt_dt


def to_np(value: pd.DataFrame | np.ndarray | pd.Series) -> np.ndarray:
    """Converts a DataFrame or a Series into a numpy array"""
    if isinstance(value, (pd.Series, pd.DataFrame)):
        return value.values.flatten()
    elif isinstance(value, np.ndarray):
        return value
    else:
        raise ValueError("Variable type not understood")


def annualised_returns(total_ret: float, n_days: int) -> float:
    """Calculates annualised cumulative return from total cumulative return"""
    return (1 + total_ret) ** (252 / n_days) - 1


def create_sharpe_ratio(returns, periods=252):
    """
    Create the Sharpe ratio for the strategy, based on a
    benchmark of zero (i.e. no risk-free rate information).

    Parameters:
    returns - A pandas Series representing period percentage returns.
    periods - Daily (252), Hourly (252*6.5), Minutely(252*6.5*60) etc.
    """
    return np.sqrt(periods) * (np.mean(returns)) / np.std(returns)


def create_drawdowns(equity_curve) -> tuple:
    """
    Calculate the largest peak-to-trough drawdown of the PnL curve
    as well as the duration of the drawdown. Requires that the
    pnl_returns is a pandas Series.

    Parameters:
    pnl - A pandas Series representing period percentage returns.

    Returns:
    drawdown, duration - Highest peak-to-trough drawdown and duration.
    """

    # Calculate the cumulative returns curve
    # and set up the High Water Mark
    # Then create the drawdown and duration series
    hwm = [0]
    eq_idx = np.arange(len(equity_curve))
    eq_val = equity_curve
    drawdown = pd.Series(index=eq_idx, dtype=np.float64)
    duration = pd.Series(index=eq_idx, dtype=np.float64)

    # Loop over the index range
    for t in range(1, len(eq_idx)):
        cur_hwm = max(hwm[t - 1], eq_val[t])
        hwm.append(cur_hwm)
        drawdown[t] = hwm[t] - eq_val[t]
        duration[t] = 0 if drawdown[t] == 0 else duration[t - 1] + 1
    idx_max_duration = np.argwhere(duration.values == duration.max()).ravel()
    return drawdown.max(), int(duration.max()), idx_max_duration[0] if len(idx_max_duration) > 0 else None


class OutputAnalyzer:
    """Class to analyze portfolio returns"""
    def __init__(self, equity_curve: pd.DataFrame | np.ndarray | pd.Series = None):
        """Returns are porcentual returns. Initialize either with pnl and initial cash or with equity curve"""
        self.equity_curve = to_np(equity_curve)
        self.initial_cash = self.equity_curve[0]
        self.returns = np.empty_like(self.equity_curve, dtype=float)
        self.returns[1:] = np.diff(self.equity_curve) / self.equity_curve[:-1]
        self.returns[0] = 0         # Fill with 0, so this means no change in pnl in t=0
        # self.cum_returns = (1 + self.returns).cumprod()
        self.__drawdown = None

    @classmethod
    def from_pnl(cls, mtm: pd.DataFrame | pd.Series | np.ndarray, initial_cash: float) -> OutputAnalyzer:
        # equity_curve = np.cumsum(to_np(pnl)) + initial_cash
        equity_curve = to_np(mtm) + initial_cash
        return OutputAnalyzer(equity_curve)

    def sharpe(self) -> float:
        ratio = create_sharpe_ratio(self.returns)
        if isinstance(ratio, float):
            return ratio
        else:
            return ratio.mean()

    @property
    def drawdown(self):
        if self.__drawdown is None:
            self.__drawdown = create_drawdowns(self.equity_curve)
        return self.__drawdown

    def drawdown_max(self):
        return self.drawdown[0]

    def drawdown_duration(self):
        return self.drawdown[1]

    def drawdown_idx_end(self):
        return self.drawdown[2]

    def drawdown_idx_start(self):
        return self.drawdown[2] - self.drawdown_duration()

    def total_return(self):
        """Total return in cash"""
        return self.equity_curve[-1] - self.equity_curve[0]

    def total_return_pct(self):
        """Total return as a percentage of initial cash"""
        return self.total_return() / self.equity_curve[0]

    def total_return_pct_annualised(self):
        return_pct = self.total_return_pct()
        return annualised_returns(return_pct, len(self.equity_curve))

    def get_stats(self):
        total_return_pct = self.total_return_pct()

        sharpe_ratio = self.sharpe()
        max_dd = self.drawdown_max()
        dd_duration = self.drawdown_duration()

        # stats = [("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
        #          ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
        #          ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
        #          ("Drawdown Duration", "%d" % dd_duration)]
        stats = [("Total Return %", total_return_pct * 100.0),
                 # ("Gross total Return %", ((gross_total_return - 1.0) * 100.0)),
                 ("Sharpe Ratio", sharpe_ratio),
                 ("Max Drawdown %", (max_dd * 100.0)),
                 ("Drawdown Duration", dd_duration),
                 ("Total Return", self.total_return())
                 ]
        return stats

    def plot(self, positions, prices, symbol, model_name):

        # Positions are plotted just for the first column (first symbol)
        df_pos = positions.iloc[:, 0]

        df_bars = prices
        df_bars = df_bars.reindex(index=df_pos.index, method='pad')

        all_signals = np.sign(df_pos)
        all_signals[np.diff(all_signals, prepend=0) == 0] = np.nan
        df_signals = all_signals.dropna().reindex(index=df_bars.index)

        fig = make_subplots(rows=3, cols=1,
                            shared_xaxes=True,
                            # vertical_spacing=0.02,
                            # specs=[[{}], [{}]],
                            subplot_titles=["Positions", "PnL", "Prices"])

        plot_chart(fig, x=df_pos.index, y=df_pos, name=symbol, row=1, col=1, symbol=symbol)
        # plot_chart(fig, x=df_pos.index, y=self.pnl, name="PnL", row=2, col=1, symbol=symbol)
        plot_chart(fig, x=df_pos.index, y=self.equity_curve, name="PnL", row=2, col=1, symbol=symbol)
        plot_maxdd(fig, df_pos.index[self.drawdown_idx_start()], df_pos.index[self.drawdown_idx_end()], row=2, col=1)

        # plot_chart(fig, x=df_pos.index, y=self.equity_curve['cash'], name="Gross", row=2, col=1, symbol=self.symbol)
        # plot_chart(fig, x=df_pos.index, y=self.equity_curve['total'], name="Total", row=2, col=1, symbol=self.symbol)
        plot_chart(fig, x=df_bars.index, y=df_bars.close, name="Close", row=3, col=1,
                   signals=df_signals, symbol=symbol)

        fig.layout.title.text = f"Model: {model_name} " \
                                f"({fmt_dt(df_pos.index[0])}-{fmt_dt(df_pos.index[-1])})" \
                                f"| Sharpe: {self.sharpe():.2%} " \
                                f"| Max drawdown: {self.drawdown_max():.2%}" \
                                f"| Duration: {self.drawdown_duration():.0f}" \
                                f"| Return: {self.total_return_pct():.2%}"
        fig.show()


if __name__ == '__main__':
    from ong_trading.vectorized.pnl import pnl_positions

    prices = [1, 1, 2, 3, 2, 1, 1, 1, 0.01, 1, 2, 3]
    prices = np.array(prices)
    initial_cash = prices[0]
    positions = np.ones_like(prices)
    pnl_values = pnl_positions(positions, prices, prices)
    # pnl_values = np.diff(prices, prepend=prices[0]) * positions
    pnl = pd.DataFrame(pnl_values, index=pd.date_range("2022-01-01", periods=len(pnl_values)), columns=["close"])
    pnls = list()
    returns = list()
    df_positions = pd.DataFrame(positions, index=pnl.index)
    df_prices = pd.DataFrame(prices, columns=["close"], index=pnl.index)
    for p in pnl, pnl.close, pnl.close.values:
        result = OutputAnalyzer.from_pnl(p, initial_cash=initial_cash)
        pnls.append(result.equity_curve)
        returns.append(result.returns)
    for test_vector, test_vector_name in zip((pnls, returns), ("Pnl", "Returns")):
        for prev, next_ in zip(test_vector[:-1], test_vector[1:]):
            assert np.allclose(prev, next_, equal_nan=True), \
                f"Vectors {test_vector_name} do not match: \n{prev}\n{next_}"
    print("Analyzer initialized OK")
    print(result.drawdown)
    print(pnl_values, pnl_values[result.drawdown_idx_start()], pnl_values[result.drawdown_idx_end()])
    # result.plot(positions=pnl.iloc[1:], prices=pnl.iloc[1:], symbol="ejemplo", model_name="modelo")
    result.plot(positions=df_positions, prices=df_prices, symbol="ejemplo", model_name="modelo")

