"""
Based on https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-V/
"""
# portfolio.py

import pandas as pd

from abc import ABC, abstractmethod
from math import floor

from ong_trading.event_driven.event import FillEvent, OrderEvent, SignalEvent, MarketEvent
from ong_trading.event_driven.performance import create_sharpe_ratio, create_drawdowns, AnalyzeOutput
from ong_trading.event_driven.utils import InstrumentType, DirectionType, plot_chart


class Portfolio(ABC):
    """
    The Portfolio class handles the positions and market
    value of all instruments at a resolution of a "bar",
    i.e. secondly, minutely, 5-min, 30-min, 60 min or EOD.
    """

    @abstractmethod
    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new orders
        based on the portfolio logic.
        """
        raise NotImplementedError("Should implement update_signal()")

    @abstractmethod
    def update_fill(self, event):
        """
        Updates the portfolio current positions and holdings
        from a FillEvent.
        """
        raise NotImplementedError("Should implement update_fill()")


class BasePortfolio(Portfolio):

    def __init__(self, bars, events, cash=7000):
        self.bars = bars
        self.events = events
        self.symbol_list = self.bars.symbol_list
        self.cash = cash
        self.positions = [{s: 0 for s in self.symbol_list}]
        self.holdings = [{s: 0 for s in self.symbol_list}]
        self.holdings[-1]['cash'] = cash
        self.holdings[-1]['total'] = cash
        self.qty = 300

    def update_signal(self, signal: SignalEvent):
        """Send an order equal to the signal minus the position"""
        if not isinstance(signal, SignalEvent):
            return
        symbol = signal.symbol
        if signal.signal_type == DirectionType.BUY:
            position = signal.strength * self.qty
        elif signal.signal_type == DirectionType.SELL:
            position = -signal.strength * self.qty
        # elif signal.signal_type == "EXIT":
        #     position = 0
        else:
            raise ValueError(f"Signal type not understood: {signal.signal_type}")

        order_qty = position - self.positions[-1][symbol]
        order_type = "MKT"

        if order_qty == 0:
            return
        else:
            order = OrderEvent(symbol, order_type, abs(order_qty),
                               DirectionType.BUY if order_qty > 0 else DirectionType.SELL)
            self.events.put(order)

    def update_fill(self, fill: FillEvent):
        if not isinstance(fill, FillEvent):
            return
        self.positions[-1][fill.symbol] += fill.position * fill.quantity
        self.positions[-1]["timestamp"] = fill.timeindex
        self.holdings[-1]["timestamp"] = fill.timeindex

    def update_timeindex(self, mkt_event: MarketEvent):
        if not isinstance(mkt_event, MarketEvent):
            return
        timestamp = self.bars.get_latest_bars(self.symbol_list[0])[0].timestamp
        self.positions.append(self.positions[-1].copy())
        self.holdings.append(self.holdings[-1].copy())
        self.holdings[-1]['timestamp'] = timestamp
        self.positions[-1]['timestamp'] = timestamp
        for s in self.symbol_list:
            # TODO: correct holdings with bid-offer
            self.holdings[-1][s] += self.positions[-1][s] * self.bars.get_latest_bars(s)[0].close

    def output_summary_stats(self):
        """
        Creates a list of summary statistics for the portfolio such
        as Sharpe Ratio and drawdown information.
        """
        df_pos = pd.DataFrame(self.positions).set_index('timestamp')
        df_hold = pd.DataFrame(self.holdings).set_index('timestamp')
        df_hold['returns'] = df_hold['total'].pct_change()
        df_hold['equity_curve'] = (1.0 + df_hold['returns']).cumprod()
        self.equity_curve = df_hold

        total_return = self.equity_curve['equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']

        sharpe_ratio = create_sharpe_ratio(returns)
        max_dd, dd_duration = create_drawdowns(pnl)

        stats = [("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
                 ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
                 ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
                 ("Drawdown Duration", "%d" % dd_duration)]

        from matplotlib import pyplot as plt
        plt.figure()
        ax = plt.subplot(2, 1, 1)
        ax.plot(df_pos)
        ax = plt.subplot(2,1,2)
        ax.plot(self.equity_curve['cash'])
        plt.show()

        return stats


        pass


# class NaivePortfolio_OLD(Portfolio):
#     """
#     The NaivePortfolio object is designed to send orders to
#     a brokerage object with a constant quantity size blindly,
#     i.e. without any risk management or position sizing. It is
#     used to test simpler strategies such as BuyAndHoldStrategy.
#     """
#
#     def __init__(self, bars, events, start_date, initial_capital=100000.0):
#         """
#         Initialises the portfolio with bars and an event queue.
#         Also includes a starting datetime index and initial capital
#         (USD unless otherwise stated).
#
#         Parameters:
#         bars - The DataHandler object with current market data.
#         events - The Event Queue object.
#         start_date - The start date (bar) of the portfolio.
#         initial_capital - The starting capital in USD.
#         """
#         self.bars = bars
#         self.events = events
#         self.symbol_list = self.bars.symbol_list
#         self.start_date = pd.Timestamp(start_date)
#         self.initial_capital = initial_capital
#
#         self.all_positions = self.construct_all_positions()
#         self.current_positions = {s: 0 for s in self.symbol_list}
#
#         self.all_holdings = self.construct_all_holdings()
#         self.current_holdings = self.construct_current_holdings()
#         self.equity_curve = None
#
#         self.signals = []       # keep record of received signals
#
#     def construct_all_positions(self):
#         """
#         Constructs the positions list using the start_date
#         to determine when the time index will begin.
#         """
#         d = {s: 0 for s in self.symbol_list}
#         d['datetime'] = self.start_date
#         return [d]
#
#     def construct_all_holdings(self):
#         """
#         Constructs the holdings list using the start_date
#         to determine when the time index will begin.
#         """
#         d = {s: 0.0 for s in self.symbol_list}
#         d['datetime'] = self.start_date
#         d['cash'] = self.initial_capital
#         d['commission'] = 0.0
#         d['total'] = self.initial_capital
#         return [d]
#
#     def construct_current_holdings(self):
#         """
#         This constructs the dictionary which will hold the instantaneous
#         value of the portfolio across all symbols.
#         """
#         d = {s: 0 for s in self.symbol_list}
#         d['cash'] = self.initial_capital
#         d['commission'] = 0.0
#         d['total'] = self.initial_capital
#         return d
#
#     def update_timeindex(self, event):
#         """
#         Adds a new record to the position matrix for the current
#         market data bar. This reflects the PREVIOUS bar, i.e. all
#         current market data at this stage is known (OLHCVI).
#
#         Makes use of a MarketEvent from the events queue.
#         """
#         bars = {}
#         for sym in self.symbol_list:
#             bars[sym] = self.bars.get_latest_bars(sym, N=1)
#
#         # Update positions
#         dp = {s: 0 for s in self.symbol_list}
#         # dp['datetime'] = bars[self.symbol_list[0]][0][1]
#         dp['datetime'] = bars[self.symbol_list[0]][0].timestamp
#
#         for s in self.symbol_list:
#             dp[s] = self.current_positions[s]
#
#         # Append the current positions
#         self.all_positions.append(dp)
#
#         # Update holdings
#         dh = {k: v for k, v in [(s, 0) for s in self.symbol_list]}
#         dh['datetime'] = bars[self.symbol_list[0]][0].timestamp
#         dh['cash'] = self.current_holdings['cash']
#         dh['commission'] = self.current_holdings['commission']
#         dh['total'] = self.current_holdings['cash']
#
#         for s in self.symbol_list:
#             # Approximation to the real value
#             # TODO: use bid/offer depending on the sign of the current position
#             # market_value = self.current_positions[s] * bars[s][0][5]
#             market_value = self.current_positions[s] * bars[s][0].close
#             dh[s] = market_value
#             dh['total'] += market_value
#
#         # Append the current holdings
#         self.all_holdings.append(dh)
#
#     def update_positions_from_fill(self, fill):
#         """
#         Takes a FillEvent object and updates the position matrix
#         to reflect the new position.
#
#         Parameters:
#         fill - The FillEvent object to update the positions with.
#         """
#         # Update positions list with new quantities
#         self.current_positions[fill.symbol] += fill.position * fill.quantity
#
#     def update_holdings_from_fill(self, fill):
#         """
#         Takes a FillEvent object and updates the holdings' matrix
#         to reflect the holdings value.
#
#         Parameters:
#         fill - The FillEvent object to update the holdings with.
#         """
#
#         # Update holdings list with new quantities
#         # fill_cost = self.bars.get_latest_bars(fill.symbol)[0][5]  # Close price
#         # TODO: this cost depends if instrument is equity or a CFD. This is Equity cost (full cost)
#         fill_cost = self.bars.get_latest_bars(fill.symbol)[0].close  # Close price
#         cost = fill.position * fill_cost * fill.quantity
#         self.current_holdings[fill.symbol] += cost
#         commission = fill.commission or 0
#         self.current_holdings['commission'] += commission
#         self.current_holdings['cash'] -= (cost + commission)
#         self.current_holdings['total'] -= (cost + commission)
#
#     def update_fill(self, event):
#         """
#         Updates the portfolio current positions and holdings
#         from a FillEvent.
#         """
#         if event.type == 'FILL':
#             self.update_positions_from_fill(event)
#             self.update_holdings_from_fill(event)
#
#     def generate_naive_order(self, signal):
#         """
#         Simply transacts an OrderEvent object as a constant quantity
#         sizing of the signal object, without risk management or
#         position sizing considerations.
#
#         Parameters:
#         signal - The SignalEvent signal information.
#         """
#         order = None
#
#         symbol = signal.symbol
#         direction = signal.signal_type
#         strength = signal.strength
#
#         mkt_quantity = floor(1000 * strength)
#         cur_quantity = self.current_positions[symbol]
#         order_type = 'MKT'
#
#         # TODO: change this behavior: the signal should give the desired position and this event_driven translate into orders
#         if direction == DirectionType.BUY and cur_quantity == 0:
#             order = OrderEvent(symbol, order_type, mkt_quantity, 'BUY')
#         elif direction == DirectionType.SELL and cur_quantity == 0:
#             order = OrderEvent(symbol, order_type, mkt_quantity, 'SELL')
#         # elif direction == 'EXIT' and cur_quantity > 0:
#         #     order = OrderEvent(symbol, order_type, abs(cur_quantity), 'SELL')
#         # elif direction == 'EXIT' and cur_quantity < 0:
#         #     order = OrderEvent(symbol, order_type, abs(cur_quantity), 'BUY')
#         return order
#
#     def update_signal(self, event):
#         """
#         Acts on a SignalEvent to generate new orders
#         based on the portfolio logic.
#         """
#         if isinstance(event, SignalEvent):
#             order_event = self.generate_naive_order(event)
#             self.events.put(order_event)
#
#     def create_equity_curve_dataframe(self):
#         """
#         Creates a pandas DataFrame from the all_holdings
#         list of dictionaries.
#         """
#         curve = pd.DataFrame(self.all_holdings)
#         curve.set_index('datetime', inplace=True)
#         curve['returns'] = curve['total'].pct_change()
#         curve['equity_curve'] = (1.0 + curve['returns']).cumprod()
#         self.equity_curve = curve
#
#     def output_summary_stats(self):
#         """
#         Creates a list of summary statistics for the portfolio such
#         as Sharpe Ratio and drawdown information.
#         """
#         if self.equity_curve is None:
#             self.create_equity_curve_dataframe()
#         total_return = self.equity_curve['equity_curve'][-1]
#         returns = self.equity_curve['returns']
#         pnl = self.equity_curve['equity_curve']
#
#         sharpe_ratio = create_sharpe_ratio(returns)
#         max_dd, dd_duration = create_drawdowns(pnl)
#
#         stats = [("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
#                  ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
#                  ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
#                  ("Drawdown Duration", "%d" % dd_duration)]
#
#         df_pos = pd.DataFrame(self.all_positions)
#         df_pos.set_index('datetime', inplace=True)
#
#         from matplotlib import pyplot as plt
#         plt.figure()
#         ax = plt.subplot(2, 1, 1)
#         ax.plot(df_pos)
#         ax = plt.subplot(2,1,2)
#         ax.plot(self.equity_curve['cash'])
#         plt.show()
#
#         return stats


class NaivePortfolio(Portfolio):
    """
    The NaivePortfolio object is designed to send orders to
    a brokerage object with a constant quantity size blindly,
    i.e. without any risk management or position sizing. It is
    used to test simpler strategies such as BuyAndHoldStrategy.
    """

    def __init__(self, broker, bars, events, start_date, instrument=InstrumentType.Stock, initial_capital=100000.0):
        """
        Initialises the portfolio with bars and an event queue.
        Also includes a starting datetime index and initial capital
        (USD unless otherwise stated).

        Parameters:
        broker - The broker that executes all orders.
        bars - The DataHandler object with current market data.
        events - The Event Queue object.
        start_date - The start date (bar) of the portfolio.
        initial_capital - The starting capital in USD.
        """
        self.broker = broker
        self.bars = bars
        self.events = events
        self.symbol_list = self.bars.symbol_list
        self.start_date = pd.Timestamp(start_date)
        self.initial_capital = initial_capital

        self.all_positions = self.construct_all_positions()
        self.current_positions = {s: 0 for s in self.symbol_list}

        self.all_holdings = self.construct_all_holdings()
        self.current_holdings = self.construct_current_holdings()
        self.equity_curve = None

        self.instrument = instrument
        if self.instrument not in InstrumentType.__members__.values():
            raise ValueError(f"Instrument {self.instrument} not valid")

        self.signals = list()       # keep record of signals

    def construct_all_positions(self):
        """
        Constructs the positions list using the start_date
        to determine when the time index will begin.
        """
        d = {s: self.broker.positions[s] for s in self.symbol_list}
        # d = {s: 0 for s in self.symbol_list}
        d['datetime'] = self.start_date
        return [d]

    def construct_all_holdings(self):
        """
        Constructs the holdings list using the start_date
        to determine when the time index will begin.
        """
        d = {s: 0.0 for s in self.symbol_list}
        d['datetime'] = self.start_date
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return [d]

    def construct_current_holdings(self):
        """
        This constructs the dictionary which will hold the instantaneous
        value of the portfolio across all symbols.
        """
        d = {s: 0 for s in self.symbol_list}
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d

    def update_timeindex(self, event: MarketEvent):
        """
        Adds a new record to the position matrix for the current
        market data bar. This reflects the PREVIOUS bar, i.e. all
        current market data at this stage is known (OLHCVI).

        Makes use of a MarketEvent from the events queue.
        """
        bars = {}
        for sym in self.symbol_list:
            bars[sym] = self.bars.get_latest_bars(sym, N=2)

        # Update positions
        # dp = {s: self.current_positions[s] for s in self.symbol_list}
        dp = {s: self.broker.positions[s] for s in self.symbol_list}
        dp['datetime'] = bars[self.symbol_list[0]][-1].timestamp

        # Append the current positions
        if dp['datetime'] == self.all_positions[-1]['datetime']:
            # replace instead of append
            self.all_positions[-1] = dp
        else:
            self.all_positions.append(dp)

        # Update holdings
        dh = dict()
        dh['datetime'] = dp['datetime']
        dh['cash'] = self.current_holdings['cash']
        dh['commission'] = self.current_holdings['commission']
        # dh['total'] = self.current_holdings['total']

        # Read total from broker
        dh['total'] = self.broker.cash

        # for s in self.symbol_list:
        #     # TODO: use bid/offer depending on the sign of the current position
        #     # Approximation to the real value
        #     if self.instrument == InstrumentType.Stock:
        #         market_value = self.current_positions[s] * bars[s][-1].close
        #     elif self.instrument == InstrumentType.CfD:
        #         if len(bars[s]) > 1:
        #             market_value = self.current_positions[s] * (bars[s][-1].close - bars[s][-2].close)
        #         else:
        #             market_value = 0
        #
        #     dh[s] = market_value
        #     dh['total'] += market_value

        # Append the current positions
        if dh['datetime'] == self.all_holdings[-1]['datetime']:
            # replace instead of append
            self.all_holdings[-1] = dh
        else:
            self.all_holdings.append(dh)

    def update_positions_from_fill(self, fill: FillEvent):
        """
        Takes a FillEvent object and updates the position matrix
        to reflect the new position.

        Parameters:
        fill - The FillEvent object to update the positions with.
        """
        self.current_positions[fill.symbol] += fill.quantity

    def update_holdings_from_fill(self, fill: FillEvent):
        """
        Takes a FillEvent object and updates the holdings' matrix
        to reflect the holdings value.

        Parameters:
        fill - The FillEvent object to update the holdings with.
        """

        # # Update holdings list with new quantities
        # # fill_cost = self.bars.get_latest_bars(fill.symbol)[0][5]  # Close price
        # # TODO: this cost depends if instrument is equity or a CFD. This is Equity cost (full cost)
        # print(fill.fill_cost)
        # fill_cost = self.bars.get_latest_bars(fill.symbol)[-1].close  # Close price
        # if self.instrument == InstrumentType.Stock:
        #     cost = fill.position * fill_cost * fill.quantity
        # elif self.instrument == InstrumentType.CfD:
        #     # TODO: CfD is hardcoded to cost just 20% of total quantity (5x leverage)
        #     cost = fill.position * fill_cost * fill.quantity * 0.2
        # else:
        #     cost = 0
        cost = fill.fill_cost
        self.current_holdings[fill.symbol] += cost
        commission = fill.commission or 0
        self.current_holdings['commission'] += commission
        self.current_holdings['cash'] -= (cost + commission)
        self.current_holdings['total'] -= (cost + commission)

    def update_fill(self, fill):
        """
        Updates the portfolio current positions and holdings
        from a FillEvent.
        """
        # if fill.type == "FILL":     #isinstance(fill, FillEvent):
        if isinstance(fill, FillEvent):
            self.update_positions_from_fill(fill)
            self.update_holdings_from_fill(fill)

    def generate_naive_order(self, signal: SignalEvent):
        """
        Simply transacts an OrderEvent object as a constant quantity
        sizing of the signal object, without risk management or
        position sizing considerations.

        Parameters:
        signal - The SignalEvent signal information.
        """
        if not isinstance(signal, SignalEvent):
            return
        if signal.datetime < self.start_date:
            return      # Do not trade before start date

        symbol = signal.symbol
        direction = signal.signal_type      # buy=1, neutral=0, sell=-1
        strength = signal.strength
        # Leave 5% of capital out for commissions, etc
        # Consider leverage
        # TODO: manage lots here
        mkt_quantity = floor(0.95 * self.initial_capital / len(self.symbol_list) /
                             self.bars.get_latest_bars(signal.symbol)[0].close * strength
                             * 0.2)     # 20% of leverage
        cur_position = self.current_positions[symbol]
        # order_type = 'MKT'
        limit_price = None      # Price limit for a market order is None

        desired_position = mkt_quantity * direction.value
        if self.instrument == InstrumentType.Stock:     # For stocks, positions cannot be negative
            desired_position = max(0, desired_position)
        order_quantity = desired_position - cur_position
        if order_quantity != 0:
            order = OrderEvent(symbol, limit_price, order_quantity, instrument=self.instrument)
            return order
        else:
            return None     # No order is sent if positions does not change

    def update_signal(self, event):
        """
        Acts on a SignalEvent to generate new orders
        based on the portfolio logic.
        """
        # if event.type == 'SIGNAL':
        if isinstance(event, SignalEvent):
            if event.datetime >= self.start_date:        # skip signals before start date
                last_signal = self.signals[-1] if self.signals else None
                if not (last_signal is not None and event.symbol in last_signal
                        and last_signal[event.symbol] == event.strength * event.signal_type.value):
                    order_event = self.generate_naive_order(event)
                    if order_event:
                        self.events.put(order_event)
                signal = {"datetime": event.datetime, event.symbol: event.strength * event.signal_type.value}
                # Only record signal changes
                if not self.signals or self.signals[-1].get(event.symbol, -1) != signal.get(event.symbol, 0):
                    self.signals.append(signal)

    def create_equity_curve_dataframe(self):
        """
        Creates a pandas DataFrame from the all_holdings
        list of dictionaries.
        """
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index('datetime', inplace=True)
        curve['returns'] = curve['total'].pct_change()
        curve['equity_curve'] = (1.0 + curve['returns']).cumprod()
        curve['gross_returns'] = curve['cash'].pct_change()
        curve['gross_equity_curve'] = (1.0 + curve['gross_returns']).cumprod()
        self.equity_curve = curve

    def output_summary_stats(self, plot=True, model_name: str = ""):
        """
        Creates a list of summary statistics for the portfolio such
        as Sharpe Ratio and drawdown information.
        :param plot: if True (default),  a plotly summary chart is shown
        :param model_name: a model name to enrich plotting
        """
        if self.equity_curve is None:
            self.create_equity_curve_dataframe()

        bars = self.bars.get_latest_bars(self.bars.symbol_list[0], N=None)
        df_bars = pd.DataFrame(bars).set_index("timestamp")
        df_pos = pd.DataFrame(self.all_positions).set_index("datetime")
        analisys = AnalyzeOutput(pnl=self.equity_curve['total'])
        stats = analisys.get_stats()
        print(stats)
        if plot:
            analisys.plot(positions=df_pos, prices=df_bars, symbol=self.symbol_list[0], model_name=model_name)
        # return stats


        total_return = self.equity_curve['equity_curve'][-1]
        gross_total_return = self.equity_curve['gross_equity_curve'][-1]
        returns = self.equity_curve['returns']
        pnl = self.equity_curve['equity_curve']
        # gross_pnl = self.equity_curve['gross_equity_curve']

        sharpe_ratio = create_sharpe_ratio(returns)
        max_dd, dd_duration, *_ = create_drawdowns(pnl)

        # stats = [("Total Return", "%0.2f%%" % ((total_return - 1.0) * 100.0)),
        #          ("Sharpe Ratio", "%0.2f" % sharpe_ratio),
        #          ("Max Drawdown", "%0.2f%%" % (max_dd * 100.0)),
        #          ("Drawdown Duration", "%d" % dd_duration)]
        stats = [("Total Return %",  ((total_return - 1.0) * 100.0)),
                 # ("Gross total Return %",  ((gross_total_return - 1.0) * 100.0)),
                 ("Sharpe Ratio",  sharpe_ratio),
                 ("Max Drawdown %",  (max_dd * 100.0)),
                 ("Drawdown Duration", dd_duration),
                 ("Total Return", returns[-1])
                 ]

        if False and plot:
            df_pos = pd.DataFrame(self.all_positions)
            df_pos.set_index('datetime', inplace=True)

            bars = self.bars.get_latest_bars(self.bars.symbol_list[0], N=None)
            df_bars = pd.DataFrame(bars).set_index("timestamp")
            df_bars = df_bars.reindex(index=df_pos.index, method='pad')
            from plotly.subplots import make_subplots
            df_signals = pd.DataFrame(self.signals, index=df_pos.index).reindex(index=df_bars.index)

            fig = make_subplots(rows=3, cols=1,
                                shared_xaxes=True,
                                # vertical_spacing=0.02,
                                # specs=[[{}], [{}]],
                                subplot_titles=["Positions", "PnL", "Prices"])

            symbol = self.symbol_list[0]

            plot_chart(fig, x=df_pos.index, y=df_pos[symbol], name=symbol, row=1, col=1, symbol=symbol)
            plot_chart(fig, x=df_pos.index, y=self.equity_curve['cash'], name="Gross", row=2, col=1, symbol=symbol)
            plot_chart(fig, x=df_pos.index, y=self.equity_curve['total'], name="Total", row=2, col=1, symbol=symbol)
            plot_chart(fig, x=df_bars.index, y=df_bars.close, name="Close", row=3, col=1,
                       signals=df_signals[symbol], symbol=symbol)

            fig.show()

        return stats
