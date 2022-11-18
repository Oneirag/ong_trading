from enum import Enum
from time import time

import numpy as np
import pandas as pd
from plotly import graph_objects as go


class InstrumentType(Enum):
    Stock = "Stock"
    CfD = "CfD"


def get_leverage(instrument: InstrumentType, symbol: str) -> int:
    """Returns leverage. For cfd will be 5. Symbol is currently not used"""
    return 1 if instrument == InstrumentType.Stock else 5


class DirectionType(Enum):
    BUY = 1
    SELL = -1
    NEUTRAL = 0


def plot_chart(fig, x, y, name: str, symbol: str, row: int, col: int, signals: pd.Series = None):
    """
    Plots a linechart in a subplot. The line segments are green when the segment increases, blue when decreased and
    black otherwise.
    If a signal is passed (a pd.Series that should range between -1 and 1), then for each -1 a red triangle is plot
    above the chart and for each +1 a green triangle is plotted bellow the chart.
    :param fig: an already created fig (with make_subplot). The function does not call show()
    :param x: x values (pandas index)
    :param y: y values (pandas series)
    :param name: name of the series (for legend)
    :param symbol: symbol of the series (for legend)
    :param row: row of the subplot
    :param col: col of the subplot
    :param signals: optional dataframe/series with the signals (+1 to buy, -1 to sell) with the same size as x
    :return: None
    """

    def to_array(arr):
        """Converts x to np.ndarray if is not a ndarray"""
        return arr if isinstance(arr, np.ndarray) else arr.values

    xl = to_array(x)
    yl = to_array(y)

    # now = time()
    # diffs = np.sign(np.diff(yl, prepend=0))
    # indexes = np.nonzero(diffs)[0] + 1
    # groups = np.split(yl, indexes)
    #
    # dif = np.sign(np.diff(yl))
    # idx = np.argwhere(np.diff(dif, prepend=0) != 0)
    #
    # for tn in range(len(xl) - 1):
    #     fig.add_trace(go.Scatter(
    #         x=xl[tn: tn + 2],
    #         y=yl[tn: tn + 2],
    #         line=dict(color="red" if yl[tn + 1] < yl[tn] else "green"),
    #         showlegend=False if tn > 0 else True,
    #         name=name
    #     ),
    #         row=row, col=col
    #     )
    # print("elapsed {:.2f}seconds".format(time()-now))

    now = time()
    dif = np.sign(np.diff(y))

    non_zero_diffs = np.diff(dif) != 0
    indexes = np.nonzero(non_zero_diffs)[0] + 1
    groups = np.split(np.arange(len(x)), indexes)  # Groups of indexes

    for group in groups:
        start_i = group[0]
        end_i = group[-1] + 2
        color = "red" if dif[start_i] < 0 else "black" if dif[start_i] == 0 else "green"
        fig.add_trace(go.Scatter(x=x[start_i:end_i], y=y[start_i:end_i],
                                 line=dict(color=color),
                                 showlegend=False if start_i > 0 else True,
                                 connectgaps=False,
                                 name=name,
                                 ), row=row, col=col)

    print("Elapsed {:.2f}seconds".format(time() - now))

    offset = 0.05
    if signals is not None:
        signals = signals * y.max() * offset
        buy_signals = y - signals.where(signals > 0)
        sell_signals = y - signals.where(signals < 0)
        neutral_signals = y - signals.where(signals == 0)
        fig.add_trace(go.Scatter(x=x, y=buy_signals, name=f"{symbol}_buy",
                                 mode="markers", marker=dict(symbol="arrow-up", size=10, color="green")),
                      row=3, col=1)
        fig.add_trace(go.Scatter(x=x, y=sell_signals, name=f"{symbol}_sell",
                                 mode="markers", marker=dict(symbol="arrow-down", size=10, color="red")),
                      row=3, col=1)

        fig.add_trace(go.Scatter(x=x, y=neutral_signals, name=f"{symbol}_neutral",
                                 mode="markers", marker=dict(symbol="diamond", size=10, color="gray")),
                      row=3, col=1)


def plot_maxdd(fig, date_start, date_end, row, col):

    # Add shape regions
    fig.add_vrect(
        x0=date_start, x1=date_end,
        fillcolor="LightSalmon", opacity=0.3,
        layer="below", line_width=0,
        row=row, col=col, name="drawdown"
    )
