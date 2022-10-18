"""
Based on https://www.quantstart.com/articles/Event-Driven-Backtesting-with-Python-Part-II/

MarketEvent - This is triggered when the outer while loop begins a new "heartbeat". It occurs when the DataHandler
    object receives a new update of market data for any symbols which are currently being tracked.
    It is used to trigger the Strategy object generating new trading signals. The event object simply contains an
    identification that it is a market event, with no other structure.
SignalEvent - The Strategy object utilises market data to create new SignalEvents. The SignalEvent contains a ticker
    symbol, a timestamp for when it was generated and a direction (long or short). The SignalEvents are utilised by
    the Portfolio object as advice for how to trade.
OrderEvent - When a Portfolio object receives SignalEvents it assesses them in the wider context of the portfolio,
    in terms of risk and position sizing. This ultimately leads to OrderEvents that will be sent to an
    ExecutionHandler.
FillEvent - When an ExecutionHandler receives an OrderEvent it must transact the order. Once an order has been
    transacted it generates a FillEvent, which describes the cost of purchase or sale as well as the transaction costs,
     such as fees or slippage.
"""

from ong_trading import logger
from ong_trading.event_driven.utils import InstrumentType


class Event(object):
    """
    Event is base class providing an interface for all subsequent
    (inherited) events, that will trigger further events in the
    trading infrastructure.
    """
    logger = logger

    def __init__(self):
        self.type = self.__class__.__name__

    def __str__(self):
        class_name = self.__class__.__name__
        return f"{class_name}: {self.__dict__}"


class MarketEvent(Event):
    """
    Handles the event of receiving a new market update with
    corresponding bars.
    """

    def __init__(self, **kwargs):
        """
        Initialises the MarketEvent.
        """
        super().__init__()
        # Copy any kwargs to self, for debugging reasons
        for k, v in kwargs.items():
            setattr(self, k, v)


class SignalEvent(Event):
    """
    Handles the event of sending a Signal from a Strategy object.
    This is received by a Portfolio object and acted upon.
    """

    def __init__(self, symbol, datetime, signal_type, strength=1):
        """
        Initialises the SignalEvent.

        Parameters:
        symbol - The ticker symbol, e.g. 'GOOG'.
        datetime - The timestamp at which the signal was generated.
        signal_type - 'LONG' or 'SHORT'.
        strength - 0 to 1, to measure signal strength
        """

        super().__init__()
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type
        # This variable was missing in the web post and NaivePortfolio would not work otherwise
        self.strength = strength


class OrderEvent(Event):
    """
    Handles the event of sending an Order to an execution system.
    The order contains a symbol (e.g. GOOG), a type (market or limit),
    quantity and a direction.
    """

    def __init__(self, symbol, limit_price, quantity, direction,
                 instrument: InstrumentType = InstrumentType.Stock):
        """
        Initialises the order type, setting whether it is
        a Market order ('MKT') or Limit order ('LMT'), has
        a quantity (integral) and its direction ('BUY' or
        'SELL').

        Parameters:
        symbol - The instrument to trade.
        limit_price - If None, it is a market order (to execute immediately), otherwise it has an execution price
        quantity - Non-negative integer for quantity.
        direction - 'BUY' or 'SELL' for long or short.
        instrument - "Stocks" or "CFD" for a physical or financial deal
        """

        super().__init__()
        self.symbol = symbol
        self.limit_price = limit_price
        self.quantity = quantity
        self.direction = direction
        self.instrument = instrument

    def print_order(self):
        """
        Outputs the values within the Order.
        """
        msg = f"Order: Symbol={self.symbol} Limit={self.limit_price}, Quantity={self.quantity}, " \
              f"Direction={self.direction}, instrument={self.instrument}"
        self.logger.info(msg)


class FillEvent(Event):
    """
    Encapsulates the notion of a Filled Order, as returned
    from a brokerage. Stores the quantity of an instrument
    actually filled and at what price. In addition, stores
    the commission of the trade from the brokerage.
    """

    def __init__(self, timeindex, symbol, exchange, quantity,
                 direction, fill_cost, price, commission=None):
        """
        Initialises the FillEvent object. Sets the symbol, exchange,
        quantity, direction, cost of fill and an optional
        commission.

        If commission is not provided, the Fill object will
        calculate it based on the trade size and Interactive
        Brokers fees.

        Parameters:
        timeindex - The bar-resolution when the order was filled.
        symbol - The instrument which was filled.
        exchange - The exchange where the order was filled.
        quantity - The filled quantity.
        direction - The direction of fill ('BUY' or 'SELL')
        fill_cost - The holdings value in dollars.
        price - The price at which deal was closed
        commission - An optional commission sent from IB.
        """

        super().__init__()
        self.timeindex = timeindex
        self.symbol = symbol
        self.exchange = exchange
        self.quantity = quantity
        self.direction = direction
        self.position = direction.value

        self.fill_cost = fill_cost
        self.price = price

        # Calculate commission
        if commission is None:
            self.commission = self.calculate_ib_commission()
        else:
            self.commission = commission

    def calculate_ib_commission(self):
        """
        Calculates the fees of trading based on an Interactive
        Brokers fee structure for API, in USD.

        This does not include exchange or ECN fees.

        Based on "US API Directed Orders":
        https://www.interactivebrokers.com/en/index.php?f=commission&p=stocks2
        """

        if self.fill_cost is None:      # In backtesting fill_cost is calculated in NaivePortfolio
            return None

        full_cost = 1.3
        if self.quantity <= 500:
            full_cost = max(1.3, 0.013 * self.quantity)
        else:  # Greater than 500
            full_cost = max(1.3, 0.008 * self.quantity)
        full_cost = min(full_cost, 0.5 / 100.0 * self.quantity * self.fill_cost)
        return full_cost


class UserNotifyEvent(Event):
    """Event that has to be notified to the user (e.g. lack of cash)
    Notification should be done in an email, Telegram bot...
    """
    def __init__(self, msg: str):
        super().__init__()
        self.msg = msg
