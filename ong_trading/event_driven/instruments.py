"""
This module contains several instruments (Stocks, CfD)
and methods for valuating them
"""
import warnings
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from dataclasses import dataclass

from ong_trading.event_driven.data import Bar
from ong_trading.event_driven.exceptions import PositionBelowMinimumException, NoCashException
from ong_trading.event_driven.rates import Rates


@dataclass
class TradeCost:
    """Stores cash spent in the trade, mark-to-market and other fees"""
    cash: float = 0
    mtm: float = 0
    others: float = 0


def calculate_trade_price(new_position: float, bid: float, offer: float) -> float:
    """Calculates trade price of a new position. If new_position<0 (sale) then trade price is bid else offer"""
    return bid if new_position < 0 else offer


class Instrument(ABC):
    """
    Base class for a generic instrument. Includes methods for trading, calculating Mark-To-Market, dividend management
    and end of day calculations
    """
    def __init__(self, min_position, rates: Rates = None):
        """
        Creates an instrument. Rates are needed to compute financing costs
        :param min_position:
        :param rates: a Rates instance, only needed for valuating overnight costs in CfD.
        """
        self.min_position = min_position
        self.position = 0
        self.trade_price = None  # Price at which instrument was closed
        self.total_dividends = 0
        self.last_day = None
        self.rates = rates
        self._total_mtm = 0
        self._total_cash = 0
        self._total_others = 0
        self._total_nominal = 0    # Total nominal value of deals

    @property
    def total_mtm(self):
        return self._total_mtm

    @property
    def total_commissions(self) -> float:
        """Returns the total of commissions, taxes, etc. It is already included in mtm and cash"""
        return self._total_others

    def calculate_new_positions(self, trade_volume) -> tuple:
        """Given a trade volume, defines if creates a new position or not, returning the pure new position and the
        closing position volume"""
        additional_volume = 0
        closing_volume = 0
        if self.position == 0:
            additional_volume = trade_volume
        elif np.sign(self.position) == np.sign(trade_volume):
            additional_volume = trade_volume
        else:
            closing_volume = np.sign(trade_volume) * min(abs(self.position), abs(trade_volume))
            additional_volume = trade_volume - closing_volume
        return additional_volume, closing_volume

    def trade(self, equity: float, trade_volume: float, bid: float, offer: float, date: pd.Timestamp) -> TradeCost:
        """
        Executes a trade and returns the cash needed to trade an instrument as a tuple. First element
        of the tuple is the market cost, second element is other costs (taxes, fees...)
        :param equity: the total cash available. If no cash available, throws
        :param trade_volume: volume of the instrument to trade (>0 for Buy, <0 for Sell)
        :param bid: market bid price
        :param offer: market offer price
        :param date: when deal was closed
        :return: the amount of cash needed to trade this Instrument (positive if spent, negative if received)
        :raises: a PositionBelowMinimumException if it cannot be traded as it would create a position bellow minimum
        """
        if trade_volume == 0:
            return TradeCost()
        self.mtm(bid, offer)    # Update mtm
        old_position = self.position        # Store old position just in case order cannot be filled
        new_position = self.position + trade_volume
        if new_position < self.min_position:
            raise PositionBelowMinimumException(f"Position {trade_volume} would create a position {new_position} "
                                                f"below its minimum of {self.min_position}")

        additional_volume, closing_volume = self.calculate_new_positions(trade_volume)
        self.position = new_position
        self.last_day = date
        trade_price = calculate_trade_price(trade_volume, bid, offer)
        comm = self.calculate_trade_commissions(trade_price, closing_volume, additional_volume)
        trade_nominal = trade_price * trade_volume + comm
        self._total_nominal += trade_nominal
        mtm = self.mtm(bid, offer)
        trade_required_cash = self.calculate_required_cash(trade_nominal, mtm)
        if trade_required_cash > equity:
            self.position = old_position
            raise NoCashException(f"No cash available. Needed {trade_required_cash:.2f} but available {equity:.2f}")
        # This is updated only if deal was closed
        if additional_volume != 0:      # Trade price is only updated with additional trades
            self.trade_price = trade_price
        return TradeCost(trade_required_cash, self._total_mtm, comm)

    @abstractmethod
    def calculate_required_cash(self, trade_nominal: float, mtm: float) -> float:
        """Updates required cash"""
        pass

    @abstractmethod
    def calculate_trade_commissions(self, trade_price: float, closing_volume: float, opening_volume: float) -> float:
        """
        Calculates trade commissions (including fees, taxes...)
        :param trade_price: the price at which trade was closed
        :param closing_volume: the volume of the total deal that reduces current position
        :param opening_volume: the volume of the total deal that increases current position
        :return: commission value (positive meaning that commission must be paid)
        """
        return 0

    def mtm(self, bid: float, offer: float) -> float:
        """
        Returns mark-to-market at current prices (including commissions)
        :param bid: bid price
        :param offer: offer (or ask) price
        :return: the mark-to-market at current price.
        """
        if self.position == 0:
            mtm = -self._total_nominal - self._total_others
        else:
            mtm = -(calculate_trade_price(-self.position, bid, offer) * -self.position +
                    self._total_nominal + self._total_others)

        mtm += self.total_dividends - self.adjust_dividends()  # Adjust dividends
        self._total_mtm = mtm
        return self._total_mtm

    @abstractmethod
    def adjust_dividends(self) -> float:
        return 0

    def add_dividend(self, dividend: float) -> None:
        if self.position != 0:
            self.total_dividends += dividend * self.position

    def new_day(self, last_bar: Bar, dividend: float = 0) -> None:
        """Do the needed processes for each new day (update overnight/monthly commissions, store dividends)"""
        self.add_dividend(dividend)
        self.mtm(last_bar.close_bid, last_bar.close_ask)

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.position=} {self.total_mtm=}"


class StockInstrument(Instrument):
    """
    Instrument for dealing with stocks. It does not allow short positions
    """

    def __init__(self):
        super().__init__(min_position=0)

    def calculate_trade_commissions(self, trade_price: float, closing_volume: float, opening_volume: float) -> float:
        return 0

    def new_day(self, last_bar: Bar, dividend: float = 0) -> None:
        super(StockInstrument, self).new_day(last_bar, dividend)
        self.last_day = last_bar.timestamp

    def adjust_dividends(self) -> float:
        return 0  # Dividends must not be adjusted for a stock

    def calculate_required_cash(self, trade_nominal: float, mtm: float) -> float:
        return trade_nominal


class StockTobinInstrument(StockInstrument):
    """
    A Stock that includes a Tobin Tax
    """

    _tobin_tax_rate = 0.002     # 0.2% of tobin tax (to be applied in stock purchases)

    def calculate_trade_commissions(self, trade_price: float, closing_volume: float, opening_volume: float) -> float:
        if opening_volume > 0:
            return opening_volume * trade_price * self._tobin_tax_rate
        else:
            return 0


class CfDInstrument(Instrument):
    """
    A contract for differences instrument (leveraged).
    Position can be either short or long, the cash needed to open the position is just the bid-offer and
    dividends are neutral (a cfd cannot make neither profit nor loss out a planned dividend yield).
    """

    def __init__(self, rates: Rates):
        """
        Creates a CfD instrument
        """
        super(CfDInstrument, self).__init__(min_position=-np.Inf, rates=rates)

    def calculate_trade_commissions(self, trade_price: float, closing_volume: float, opening_volume: float) -> float:
        """CfD has no commission by default. If needed, it should be changed in a child class"""
        return 0    # No trade commission

    def calculate_required_cash(self, trade_nominal: float, mtm: float) -> float:
        """Returns just mark-to-market (which at trade inception is just the bid-offer spread) is required"""
        return mtm

    def adjust_dividends(self) -> float:
        """CfD must remove any effect in mtm of dividends, so this adjust is the total amount of dividends received"""
        return self.total_dividends  # For a CfD dividends must be removed


class IgCfDInstrument(CfDInstrument):
    """
    CfD with the overnight cost of IG markets broker
    """

    _default_sofr = 0.0124          # Default sofr in case rates could not be downloaded
    _lending_cost = 0.006           # Lending cost (it could vary over time, but a fixed value is used here)

    def new_day(self, last_bar: Bar, dividend: float = 0) -> None:
        """Computes overnight cost for open positions as calculated by IG Markets"""
        if self.position != 0:
            nights = (self.last_day - last_bar.timestamp).days
            position = self.position
            price = last_bar.close  # TODO: consider bid/ask, but this can be a good enough solution
            if self.rates is None:
                warnings.warn("No rates found: Falling back to 1.24%")
                sofr = self._default_sofr
            else:
                sofr = self.rates.sofr[last_bar.timestamp]
            self._total_others += nights * position * price * (0.025 - sofr) / 360
            self._total_others += nights * position * price * self._lending_cost / 360
        self.last_day = last_bar.timestamp
        super(CfDInstrument, self).new_day(last_bar)
