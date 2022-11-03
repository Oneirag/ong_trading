"""
Based on https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-VII/
"""
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
from ong_trading.event_driven.utils import plot_chart, plot_maxdd


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
    eq_idx = equity_curve.index
    eq_val = equity_curve.values
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


class AnalyzeOutput:
    """Class to analyze portfolio returns"""
    def __init__(self, pnl):
        self.pnl = pnl
        self.returns = self.pnl.pct_change()
        self.__drawdown = None

    def sharpe(self) -> float:
        ratio = create_sharpe_ratio(self.returns)
        if isinstance(ratio, float):
            return ratio
        else:
            return ratio.mean()

    @property
    def drawdown(self):
        if self.__drawdown is None:
            self.__drawdown = create_drawdowns((1 + self.returns).cumprod())
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
        return self.pnl.iloc[-1] - self.pnl.iloc[0]

    def total_return_pct(self):
        """Total return as a percentage of initial cash"""
        retval = self.total_return() / self.pnl.iloc[0]
        if isinstance(retval, float):
            return retval
        else:
            return retval.mean()

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
                 ("Total Return", self.returns[-1])
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
        plot_chart(fig, x=df_pos.index, y=self.pnl, name="PnL", row=2, col=1, symbol=symbol)
        plot_maxdd(fig, df_pos.index[self.drawdown_idx_start()], df_pos.index[self.drawdown_idx_end()], row=2, col=1)

        # plot_chart(fig, x=df_pos.index, y=self.equity_curve['cash'], name="Gross", row=2, col=1, symbol=self.symbol)
        # plot_chart(fig, x=df_pos.index, y=self.equity_curve['total'], name="Total", row=2, col=1, symbol=self.symbol)
        plot_chart(fig, x=df_bars.index, y=df_bars.close, name="Close", row=3, col=1,
                   signals=df_signals, symbol=symbol)

        date_format = "%d%b%y"      # E.g. 03Nov22
        fig.layout.title.text = f"Model: {model_name} " \
                                f"({df_pos.index[0].strftime(date_format)}-{df_pos.index[-1].strftime(date_format)})" \
                                f"| Sharpe: {self.sharpe():.2%} " \
                                f"| Max drawdown: {self.drawdown_max():.2%}" \
                                f"| Duration: {self.drawdown_duration():.0f}" \
                                f"| Return: {self.total_return_pct():.2%}"
        fig.show()


if __name__ == '__main__':
    pnl_values = [1, 1, 2, 3, 2, 1, 1, 1, 2, 3]
    pnl = pd.DataFrame(pnl_values, index=pd.date_range("2022-01-01", periods=len(pnl_values)), columns=["close"])
    result = AnalyzeOutput(pnl)
    print(result.drawdown)
    print(pnl_values, pnl_values[result.drawdown_idx_start()], pnl_values[result.drawdown_idx_end()])
    result.plot(positions=pnl.iloc[1:], prices=pnl.iloc[1:], symbol="ejemplo", model_name="modelo")

