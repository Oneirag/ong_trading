import pandas as pd
from time import time

from ong_trading.helpers import plot_chart

if __name__ == '__main__':
    import numpy as np
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        # vertical_spacing=0.02,
                        # specs=[[{}], [{}]],
                        subplot_titles=["Positions", "PnL", "Prices"])

    symbol = "ejemplo"
    x = np.arange(15).astype(float)
    np.random.seed(1)
    y = (np.random.rand(x.size) * 100).astype(int)
    df_signals = pd.Series(np.array([-1 if d % 3 == 0 else 1 if d % 3 == 1 else np.NAN for d in y]), index=x)
    # Try to plot segments in one color if positive
    df_pnl = pd.Series(np.array([-1 if d % 3 == 0 else 1 if d % 3 == 1 else 0 for d in y]), index=x)

    plot_chart(fig, x, df_pnl, symbol=symbol, name="pnl", row=2, col=1)
    plot_chart(fig, x, y, name="Close", symbol=symbol, row=3, col=1, signals=df_signals)

    # fig.add_trace(go.Scatter(x=df_signals.index, y=df_pnl, name=f"pnl",
    #                          mode="lines+markers",
    #                          marker=dict(symbol="arrow-down", size=10, color="red")
    #                          ),
    #               row=2, col=1)

    now = time()
    fig.show()
    print("ellapsed plotting: {:.2f}secs".format(time() - now))
