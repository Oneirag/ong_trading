"""
Class to wrap basic backtesting
"""
from itertools import product
from typing import Type

from ong_trading.event_driven.data import DataHandler
from ong_trading.event_driven.portfolio import Portfolio, NaivePortfolio
from ong_trading.strategy.strategy import Strategy
from ong_trading.event_driven.execution import ExecutionHandler, SimulatedBroker
from ong_trading.event_driven.event import FillEvent, OrderEvent, SignalEvent, MarketEvent, UserNotifyEvent
from ong_trading.helpers.overload import singledispatch
from ong_trading import logger
from queue import Queue, Empty
from ong_trading.event_driven.utils import InstrumentType
import pandas as pd


class Backtester:

    def __init__(self, data_class: Type[DataHandler],
                 portfolio_class: Type[NaivePortfolio],
                 strategy_class: Type[Strategy],
                 broker_class: Type[SimulatedBroker],
                 symbol_list,
                 cash: float,
                 instrument_type: InstrumentType,
                 start_date: pd.Timestamp):
        self.events = Queue()
        self.cash = cash

        # bars = YahooHistoricalData(events=None, symbol_list=["ELE.MC"])
        # broker = SimulatedBroker(bars=None, events=None, cash=cash)
        # port = NaivePortfolio(broker=None, bars=None, events=None, start_date="2010-01-01",
        #                       instrument=InstrumentType.CfD,
        #                       initial_capital=cash)
        # strategy = MACrossOverStrategy(bars=None, events=None, short=3, long=7)

        self.symbol_list = symbol_list
        self.instrument_type = instrument_type
        self.start_date = start_date
        self.data_class = data_class
        self.broker_class = broker_class
        self.portfolio_class = portfolio_class
        self.strategy_class = strategy_class
        self.data = self.data_class(self.events, self.symbol_list)
        self.broker = None
        self.portfolio = None
        self.strategy = None
        # Keep records of some events
        self.dates = None
        self.fill_events = None
        self.signal_events = None

    def __init_objects(self, init_broker=True, init_portfolio=True, init_strategy=True, strategy_args=None,
                       strategy_kwargs=None):
        if init_broker and self.broker_class:
            self.broker = self.broker_class(self.data, self.events, self.cash)
        if init_portfolio and self.portfolio_class:
            self.portfolio = self.portfolio_class(self.broker, self.data, self.events,
                                                  start_date=pd.Timestamp(self.start_date),
                                                  instrument=self.instrument_type,
                                                  initial_capital=self.cash)
        if init_strategy and self.strategy_class:
            self.strategy = self.strategy_class(self.data, self.events, *(strategy_args or list()),
                                                **(strategy_kwargs or dict()))
        self.signal_events = list()
        self.fill_events = list()
        self.dates = list()

    @singledispatch(MarketEvent)
    def process_event(self, event: MarketEvent):
        if self.strategy:
            self.strategy.calculate_signals(event)
        if self.broker:
            self.broker.update_timeindex(event)
        if self.portfolio:
            self.portfolio.update_timeindex(event)
        if self.data.continue_backtest:
            self.dates.append(event.timestamp)

    @singledispatch(SignalEvent)
    def process_event(self, event: SignalEvent):
        # Portfolio transforms signals into orders
        if self.portfolio:
            self.portfolio.update_signal(event)
        self.signal_events.append(event)

    @singledispatch(OrderEvent)
    def process_event(self, event: OrderEvent):
        if self.broker:
            self.broker.execute_order(event)

    @singledispatch(FillEvent)
    def process_event(self, event: FillEvent):
        if self.portfolio:
            self.portfolio.update_fill(event)
        self.fill_events.append(event)

    @singledispatch(UserNotifyEvent)
    def process_event(self, event: UserNotifyEvent):
        # TODO: implement a better user notification (e.g. use a Telegram bot, send an email...)
        print(event)

    def run(self, print_debug_msg: bool = False, strategy_args=None, strategy_kwargs=None,
            bars_from: int = 0, bars_to: int = -1) -> list:
        """
        Runs the trading environment and returns a list of tuples with the result of the event_driven
        :param print_debug_msg: if true, debug messages are printed and the result of the strategy is plotted
        :param strategy_args: an optional list of args to create new strategy for this event_driven
        :param strategy_kwargs: an optional dict of args to create new strategy for this event_driven
        :param bars_from: number of bars to skip for the test (default = 0)
        :param bars_to: number of bars for stopping the run (default = all)
        :return: see output of self.portfolio.output_summary_stats()
        """
        self.events.empty()  # resets queue
        self.data.reset()  # resets bars
        self.__init_objects(strategy_kwargs=strategy_kwargs, strategy_args=strategy_args)
        # Handle the events
        n_bars = 0
        while True:
            if self.data.continue_backtest:
                self.data.update_bars()
                if n_bars < bars_from:
                    self.events.empty()
                n_bars += 1
                if n_bars > bars_to:
                    break
            else:
                break
            while True:
                try:
                    event = self.events.get(False)
                except Empty:
                    break
                else:
                    if event is not None:
                        if print_debug_msg:
                            logger.debug(event)
                        self.process_event(event)
        if self.portfolio:
            return self.portfolio.output_summary_stats(plot=print_debug_msg)
        else:
            return None

    def get_signals(self) -> pd.DataFrame:
        """Gets a df of the signals emitted for the strategy for each symbol"""
        df_res = pd.DataFrame(index=self.dates)
        for signal in self.signal_events:
            df_res.loc[signal.datetime, signal.symbol] = signal.signal_type.value
        # Forward fill (signals are a trigger and don't repeat) and trim at the stating date
        df_res = df_res.ffill()[self.start_date:]
        return df_res

    def optimize(self, strategy_params, bars_from=0, bars_to=-1) -> pd.DataFrame:
        retval = dict()
        for params in strategy_params:
            if isinstance(params, (list, tuple)):
                res = self.run(print_debug_msg=False, strategy_args=params, bars_from=bars_from, bars_to=bars_to)
            else:
                res = self.run(print_debug_msg=False, strategy_kwargs=params, bars_from=bars_from, bars_to=bars_to)
            dict_result = {k: v for k, v in res}
            retval[params] = dict_result
        df_results = pd.DataFrame(retval).T
        return df_results

    def optimize_walkforward(self, strategy_params, bars_train, bars_walkforward, bars_oos):
        bars_from = 0
        bars_to = bars_train
        n_bars = self.data.n_bars - bars_oos
        return_walkforward = 0
        while bars_to < n_bars:
            # First: train and get optimized params
            # TODO: what if no signal was created in the training period...?
            results = self.optimize(strategy_params, bars_from, bars_to)
            # Get the best params, by sorting by Sharpe Ratio
            best_params = results.sort_values(by="Sharpe Ratio").index[-1]
            # Evaluate the pnl of walk forward part
            # TODO: does not keep track of previous positions
            results_walkforward = self.run(strategy_args=best_params, bars_from=bars_to + 1,
                                           bars_to=bars_to + bars_walkforward)
            return_walkforward += results_walkforward[-1][-1]
            # TODO: this waklforward overlaps training with walkforward
            bars_from += bars_train
            bars_to += bars_train
        # TODO: evaluate OOS performance
        results_oos = self.run(strategy_args=best_params, bars_from=bars_to + 1,
                               bars_to=n_bars)
        return results_oos


class SignalAnalysisBacktester(Backtester):
    """
    Slim backtesting class allowing to analyze just the information of the signals of the strategies
    """

    def __init__(self, data_class: Type[DataHandler],
                 strategy_class: Type[Strategy],
                 symbol_list,
                 start_date: pd.Timestamp):
        super().__init__(data_class=data_class, portfolio_class=None,
                         strategy_class=strategy_class, broker_class=None,
                         symbol_list=symbol_list, cash=1e7, instrument_type=None,
                         start_date=start_date)


if __name__ == '__main__':
    from ong_trading.event_driven.data import YahooHistoricalData
    from ong_trading.strategy.strategy import MACrossOverStrategy
    from ong_trading.event_driven.portfolio import NaivePortfolio
    from ong_trading.event_driven.execution import SimulatedBroker
    from ong_trading.event_driven.event import MarketEvent, SignalEvent, OrderEvent, FillEvent, UserNotifyEvent

    bt = Backtester(
        data_class=YahooHistoricalData,
        portfolio_class=NaivePortfolio,
        strategy_class=MACrossOverStrategy,
        broker_class=SimulatedBroker,
        cash=7000,
        symbol_list=["ELE.MC"],
        instrument_type=InstrumentType.CfD,
        # start_date="2022-01-01",
        start_date=pd.Timestamp("2000-01-01"),
    )

    # result = bt.run(print_debug_msg=True, strategy_kwargs=dict(short=9, long=40))
    # print(result)

    strategy_params = list(product(range(3, 10, 2), range(20, 100, 20)))

    bt.optimize_walkforward(strategy_params=strategy_params, bars_train=200, bars_walkforward=100, bars_oos=100)

    df_res = bt.optimize(strategy_params)
    print(df_res.sort_values("Sharpe Ratio"))
