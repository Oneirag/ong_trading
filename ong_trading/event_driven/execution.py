"""
Based on https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-VI/
"""
# execution.py

import datetime

from abc import ABC, abstractmethod

from .event import FillEvent, OrderEvent, MarketEvent, UserNotifyEvent
from ong_trading.event_driven.utils import InstrumentType
from ong_trading.event_driven.data import DataHandler
from ong_trading import logger


class ExecutionHandler(ABC):
    """
    The ExecutionHandler abstract class handles the interaction
    between a set of order objects generated by a Portfolio and
    the ultimate set of Fill objects that actually occur in the
    market.

    The handlers can be used to subclass simulated brokerages
    or live brokerages, with identical interfaces. This allows
    strategies to be backtested in a very similar manner to the
    live trading engine.
    """

    logger = logger

    def log_error(self, reason: str, order: OrderEvent = None):
        order_txt = f" in order {order.print_order()}" if order is not None else ""
        self.logger.error(f"Error {order_txt}: {reason}")

    def log_not_filled_order(self, order: OrderEvent, reason: str):
        """
        Logs in error that an order couldn't be filled
        :param order: the rejected order
        :param reason: an informative message for rejection
        :return:
        """
        self.log_error(reason, order)

    @abstractmethod
    def execute_order(self, event):
        """
        Takes an Order event and executes it, producing
        a Fill event that gets placed onto the Events queue.

        Parameters:
        event - Contains an Event object with order information.
        """
        raise NotImplementedError("Should implement execute_order()")


class SimulatedExecutionHandler(ExecutionHandler):
    """
    The simulated execution handler simply converts all order
    objects into their equivalent fill objects automatically
    without latency, slippage or fill-ratio issues.

    This allows a straightforward "first go" test of any strategy,
    before implementation with a more sophisticated execution
    handler.
    """

    exchange_name = "ARCA"      # Just for backtesting purposes, just a placeholder

    def __init__(self, events):
        """
        Initialises the handler, setting the event queues
        up internally.

        Parameters:
        events - The Queue of Event objects.
        """
        self.events = events

    def execute_order(self, event: OrderEvent):
        """
        Simply converts Order objects into Fill objects naively,
        i.e. without any latency, slippage or fill ratio problems.

        Parameters:
        event - Contains an Event object with order information.
        """
        if isinstance(event, OrderEvent):
            fill_event = FillEvent(datetime.datetime.utcnow(), event.symbol,
                                   self.exchange_name, event.quantity, event.direction, None)
            self.events.put(fill_event)


class SimulatedBroker(ExecutionHandler):

    def __init__(self, bars: DataHandler, events, cash):
        """
        Creates a new broker
        :param bars: market data
        :param events: events queue (to notify closed deals)
        :param cash: initial cash sent to the broker
        """
        self.bars = bars
        self.events = events
        self.exchange_name = "oscar"
        self.positions = {s: 0 for s in bars.symbol_list}
        self.cash = cash
        self.initial_cash = cash

        # TODO: management of working orders
        self.working_orders = list()
        self.trade_history = {s: list() for s in bars.symbol_list}
        self.has_positions = False

    def value_pnl_trades(self, symbol, price) -> float:
        """Returns pnl of trades of a symbol at a certain price"""
        # TODO: valuate open position at bid or offer depending on LONG or SHORT
        pnl = sum(trade.quantity * trade.position * (price - trade.price)
                  for trade in self.trade_history[symbol])
        return pnl

    def update_timeindex(self, event: MarketEvent):
        """Updates cash, processed dividends and splits,
        and makes sure that positions are not closed due to negative cash"""
        # TODO: manage bid_offer
        self.execute_working_orders()
        if not self.has_positions:
            return
        worse_pnl = 0
        close_pnl = 0
        for symbol in self.bars.symbol_list:
            last_bar = self.bars.get_latest_bars(symbol, N=1)[-1]
            position = self.positions[symbol]
            if position > 0:
                worse_price = last_bar.low
            else:
                worse_price = last_bar.high
            worse_pnl += self.value_pnl_trades(symbol, worse_price)
            close_pnl += self.value_pnl_trades(symbol, last_bar.close)
        if worse_pnl + self.initial_cash < 0:
            # Close all positions due to lack of cash (for CfD)
            # It is very extreme as simulates that high price in short positions happens at the same time as
            # low prices in long positions
            close_event = UserNotifyEvent("Closing positions due to negative cash")
            self.log_error(close_event.msg)
            self.events.put(close_event)
            self.positions = {s: 0 for s in self.bars.symbol_list}
            self.cash = 0
            self.has_positions = False
            return

        self.cash = close_pnl + self.initial_cash
        # self.logger.debug(f"Available cash for {last_bar.timestamp.isoformat()}: {self.cash}")

    def execute_order(self, event: OrderEvent):
        """Sends order to working orders internal queue. If executed, a FillEvent is sent to event queue"""
        self.working_orders.append(event)
        self.execute_working_orders()

    def execute_working_orders(self):
        """
        Tries to execute the orders in working orders calculating costs. Treat stock and CfDs in a different way
        :return: None
        """
        for idx, order in enumerate(self.working_orders):
            if self.cash == 0:
                # With no cash available it is imposible to open new positions
                self.log_not_filled_order(order, "No cash available")
                return
            current_bar = self.bars.get_latest_bars(order.symbol)[-1]
            if order.limit_price is None:   # Market order
                # TODO: implement bid-offer
                trade_price = current_bar.close  # Orders are supposed to be closed at close price
            else:
                # Check if limits make order to be executed
                # TODO: implement limit order execution
                self.log_error("Limit order not implemented", order=order)
                continue
            commission = trade_price * 0.01  # TODO: improve commission calculation
            position = self.positions.get(order.symbol, 0)
            new_position = position + order.direction.value * order.quantity
            cost = None
            if order.instrument == InstrumentType.Stock:
                if new_position < 0:
                    new_position = position  # Prevent negative positions for stocks (order rejected)
                    self.log_not_filled_order(order, "Prevent short stock position")
                else:
                    cost = order.quantity * trade_price
            elif order.instrument == InstrumentType.CfD:
                cost = 0  # TODO: No cash cost for a CfD (indeed, take bid offer into account)
            if cost is not None:
                if self.cash > cost + commission:
                    fill_event = FillEvent(timeindex=current_bar.timestamp,
                                           symbol=order.symbol,
                                           exchange=self.exchange_name,
                                           quantity=order.quantity,
                                           direction=order.direction,
                                           fill_cost=cost,
                                           price=trade_price,
                                           commission=commission)
                    self.positions[order.symbol] = new_position
                    self.events.put(fill_event)
                    self.trade_history[fill_event.symbol].append(fill_event)
                    self.cash -= fill_event.fill_cost + commission  # reduces cash
                    self.has_positions = True
                    # Remove order from queue
                    self.working_orders.pop(idx)
                else:
                    self.log_not_filled_order(order, "Insufficient cash position to negotiate")
            pass