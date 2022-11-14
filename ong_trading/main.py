"""
Based on https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-I/
"""
from queue import Queue, Empty
from itertools import product
import pandas as pd
from ong_utils import OngTimer


from ong_trading.event_driven.data import YahooHistoricalData
from ong_trading.strategy.strategy import MACrossOverStrategy
from ong_trading.event_driven.portfolio import NaivePortfolio
from ong_trading.event_driven.execution import SimulatedBroker
from ong_trading.event_driven.utils import InstrumentType
from ong_trading.event_driven.event import MarketEvent, SignalEvent, OrderEvent, FillEvent, UserNotifyEvent

events = Queue()

cash = 7000


params = list(product(range(3, 10), range(20, 100, 10)))
params = list(product(range(3, 10, 2), range(20, 100, 20)))
# params = [(8, 40)]
results = dict()

bars = YahooHistoricalData(events, ["ELE.MC"])
for param in params:
    short, long = param
    # Declare the components with respective parameters
    # strategy = Strategy(..)
    # strategy = BuyAndHoldStrategy(bars, events)
    # broker = ExecutionHandler(..)
    # broker = SimulatedExecutionHandler(events)
    broker = SimulatedBroker(bars, events, cash=cash)
    # port = Portfolio(..)
    port = NaivePortfolio(broker, bars, events, start_date="2010-01-01", instrument=InstrumentType.CfD,
                          initial_capital=cash)
    # port = NaivePortfolio(bars, events, start_date="2022-01-01", instrument=InstrumentType.Stock)
    # port = BasePortfolio(bars, events)
    strategy = MACrossOverStrategy(bars, events, short=short, long=long)
    with OngTimer(msg=f"Iteration for {param}"):
        while True:
            # Update the bars (specific event_driven event_driven, as opposed to live trading)
            if bars.continue_backtest:
                bars.update_bars()
            else:
                break
            # Handle the events
            while True:
                try:
                    event = events.get(False)
                except Empty:
                    break
                else:
                    if event is not None:
                        # logger.debug(event)
                        if isinstance(event, MarketEvent):
                            strategy.calculate_signals(event)
                            broker.update_timeindex(event)
                            port.update_timeindex(event)
                        elif isinstance(event, SignalEvent):
                            pass
                            # Portfolio transforms signals into orders
                            port.update_signal(event)
                        elif isinstance(event, OrderEvent):
                            pass
                            broker.execute_order(event)
                        elif isinstance(event, FillEvent):
                            pass
                            port.update_fill(event)
                        elif isinstance(event, UserNotifyEvent):
                            # TODO: implement a better user notification (e.g. use a Telegram bot, send an email...)
                            print(event)
            # 10-Minute heartbeat
            # time.sleep(10 * 60)
        print(param)
        result = port.output_summary_stats(plot=len(params) == 1)
        print(result)
        dict_result = {k: v for k, v in result}
        results[param] = dict_result
        bars.reset()
        events.queue.clear()

print("Done!")
df_results = pd.DataFrame(results).T
print(df_results.sort_values("Sharpe Ratio"))
