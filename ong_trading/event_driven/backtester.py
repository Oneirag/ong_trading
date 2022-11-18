"""
Class to wrap basic backtesting
"""
from __future__ import annotations

from time import time
from itertools import product
from typing import Type


from queue import Queue, Empty
import pandas as pd
from ong_utils import OngTimer
from ong_trading import logger
from ong_trading.event_driven.data import DataHandler, HistoricCSVDataHandler, YahooHistoricalData
from ong_trading.event_driven.portfolio import NaivePortfolio
from ong_trading.strategy.strategy import Strategy
from ong_trading.event_driven.execution import SimulatedBroker
from ong_trading.event_driven.event import FillEvent, OrderEvent, SignalEvent, MarketEvent, UserNotifyEvent, \
    BacktestingEndEvent, OutOfCashEvent
from ong_trading.helpers.overload import singledispatch
from ong_trading.event_driven.utils import InstrumentType


class Backtester:

    def __init__(self, cash: float, symbol_list: str | list | tuple, strategy_class: Type[Strategy],
                 start_date: pd.Timestamp | str = None, end_date: pd.Timestamp | str = None,
                 data_class: Type[HistoricCSVDataHandler] = YahooHistoricalData,
                 portfolio_class: Type[NaivePortfolio] = NaivePortfolio,
                 broker_class: Type[SimulatedBroker] = SimulatedBroker,
                 instrument_type: InstrumentType = InstrumentType.CfD,
                 strategy_args: list = None, strategy_kwargs: dict = None):
        """
        Initializes an event-driven backtester
        :param cash: initial amount of cash
        :param symbol_list: name (or list of names) of the ticker(s) to use for backtesting
        :param strategy_class: class to use for the strategy
        :param start_date: start date of the backtest (defaults to None, all data available)
        :param start_date: end date of the backtest (defaults to None, all data available)
        :param data_class: class for collecting data. Defaults to YahooHistoricalData
        :param portfolio_class: portfolio class for backtesting. Defaults to NaivePortfolio
        :param broker_class: broker class for backtesting. Defaults to SimulatedBroker
        :param instrument_type: instrument type for backtesting. Defaults to InstrumentType.CfD
        :param strategy_args: list of additional *args for strategy_class constructor
        :param strategy_kwargs: dict of additional **kwargs for strategy_class constructor

        """
        self.events = Queue()
        self.cash = cash

        self.symbol_list = symbol_list
        self.instrument_type = instrument_type
        self.start_date = pd.Timestamp(start_date) if start_date is not None else None
        self.end_date = pd.Timestamp(end_date) if end_date is not None else None
        self.data_class = data_class
        self.broker_class = broker_class
        self.portfolio_class = portfolio_class
        self.strategy_class = strategy_class
        self.strategy_args = strategy_args
        self.strategy_kwargs = strategy_kwargs
        self.data = self.data_class(self.events, self.symbol_list, start_date=self.start_date, end_date=self.end_date)
        # If start_date is None, update start_data to the one in data
        if self.start_date is None:
            self.start_date = self.data.start_date
        self.broker = None
        self.portfolio = None
        self.strategy = None
        # Keep records of some events
        self.dates = None
        self.fill_events = None
        self.signal_events = None
        self.last_processed_timestamp = None
        self.timer = OngTimer(logger=logger)
        self.timer_name = "Backtesting"
        self.has_cash = True

    def reset(self):
        """Needed for running tests on this class"""
        self.__init_objects()

    def __init_objects(self, init_broker=True, init_portfolio=True, init_strategy=True,
                       strategy_args=None,
                       strategy_kwargs=None):
        self.has_cash = True
        self.last_processed_timestamp = None
        if init_broker and self.broker_class:
            self.broker = self.broker_class(self.data, self.events, self.cash)
        if init_portfolio and self.portfolio_class:
            self.portfolio = self.portfolio_class(self.broker, self.data, self.events,
                                                  start_date=pd.Timestamp(self.start_date),
                                                  instrument=self.instrument_type,
                                                  initial_capital=self.cash)
        if init_strategy and self.strategy_class:
            self.strategy = self.strategy_class(self.data, self.events,
                                                *(strategy_args or self.strategy_args or list()),
                                                **(strategy_kwargs or self.strategy_kwargs or dict()))
        self.signal_events = list()
        self.fill_events = list()
        self.dates = list()

    @singledispatch(MarketEvent)
    def process_event(self, event: MarketEvent):
        try:
            if (self.last_processed_timestamp or event.timestamp).year != event.timestamp.year:
                print(f"Processing year: {event.timestamp.year}")
            self.last_processed_timestamp = event.timestamp
        except:
            pass
        if self.strategy and self.has_cash:
            self.strategy.calculate_signals(event)
        if self.broker:
            self.broker.update_timeindex(event)
        if self.portfolio:
            self.portfolio.update_timeindex(event)
        if self.data.continue_backtest:
            self.dates.append(event.timestamp)

    @singledispatch(OutOfCashEvent)
    def process_event(self, event: OutOfCashEvent):
        self.has_cash = False

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

    @singledispatch(BacktestingEndEvent)
    def process_event(self, event: BacktestingEndEvent):
        self.timer.toc(self.timer_name)

    def run(self, print_debug_msg: bool = False, strategy_args=None, strategy_kwargs=None,
            bars_from: int = 0, bars_to: int = -1) -> Type[NaivePortfolio]:
        """
        Runs the trading environment and returns a list of tuples with the result of the event_driven
        :param print_debug_msg: if true, debug messages are printed and the result of the strategy is plotted
        :param strategy_args: an optional list of args to create new strategy for this event_driven
        :param strategy_kwargs: an optional dict of args to create new strategy for this event_driven
        :param bars_from: number of bars to skip for the test (default = 0)
        :param bars_to: number of bars for stopping the run (default = all)
        :return: the portfolio object, so you can use portfolio.output_summary_stats() to get stats
        """
        self.timer.tic(self.timer_name)
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
                if n_bars > bars_to > -1:
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
            return self.portfolio
            # return self.portfolio.output_summary_stats(plot=print_debug_msg)
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
        super().__init__(cash=1e7, symbol_list=symbol_list, strategy_class=strategy_class, start_date=start_date,
                         data_class=data_class, portfolio_class=None, broker_class=None, instrument_type=None)


if __name__ == '__main__':
    from ong_trading.event_driven.data import YahooHistoricalData
    from ong_trading.strategy.strategy import MachineLearningStrategy, MACrossOverStrategy
    from ong_trading.event_driven.portfolio import NaivePortfolio
    from ong_trading.event_driven.execution import SimulatedBroker
    from ong_trading.event_driven.event import MarketEvent, SignalEvent, OrderEvent, FillEvent, UserNotifyEvent
    from ong_trading.features.preprocess import RLPreprocessorClose
    from ong_trading.ML.RL.config import ModelConfig

    bt = Backtester(cash=7000, symbol_list=ModelConfig.ticker, strategy_class=MachineLearningStrategy,
                    start_date=pd.Timestamp(ModelConfig.train_split_date), data_class=YahooHistoricalData,
                    portfolio_class=NaivePortfolio, broker_class=SimulatedBroker, instrument_type=InstrumentType.CfD,
                    strategy_kwargs=dict(model_path=ModelConfig.model_path(True),
                                         preprocessor=RLPreprocessorClose.load(
                                             ModelConfig.model_path("preprocessor"))
                                         ))

    # logger.setLevel(logging.INFO)
    result = bt.run(print_debug_msg=False)
    print(result.output_summary_stats(plot=True))
    exit(0)

    strategy_params = list(product(range(3, 10, 2), range(20, 100, 20)))

    bt.optimize_walkforward(strategy_params=strategy_params, bars_train=200, bars_walkforward=100, bars_oos=100)

    df_res = bt.optimize(strategy_params)
    print(df_res.sort_values("Sharpe Ratio"))
