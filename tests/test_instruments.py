"""
Tests for instruments: check correct bid-offer application, cost calculations...
"""
import unittest
from time import time

import numpy as np
import pandas as pd
from queue import Queue

from ong_trading.event_driven.instruments import CfDInstrument, StockInstrument, StockTobinInstrument
from ong_trading.event_driven.exceptions import PositionBelowMinimumException
from ong_trading.event_driven.data import GeneratedHistoricalData
from ong_trading.vectorized.pnl import pnl_positions, calculate_entry_points_vectorized, calculate_entry_points
from functools import partial


class TestInstruments(unittest.TestCase):
    def setUp(self) -> None:
        self.equity = 0
        self.equity = 10000
        self.bid = 40, 41, 39, 37, 45
        self.bid_offer = 1
        # self.bid_offer = 0
        self.ask = [b + self.bid_offer for b in self.bid]
        # dividends of 1 each 3 days
        self.dividends = [1 if d % 3 == 1 else 0 for d in range(len(self.bid))]
        # First date is computed as 3 business days ago to minimize the risk of bank holidays
        first_date = pd.Timestamp.now() - pd.tseries.offsets.BDay(3)
        self.dates = pd.date_range(first_date, periods=len(self.bid), freq="B", normalize=True)

        # Setup to create real bars
        self.symbol_list = ['fake']
        self.df = pd.DataFrame({"open": self.bid, "high": self.bid, "low": self.bid,
                                "close": self.bid,
                                "dividends": self.dividends}, index=self.dates)

        df_no_divividends = pd.DataFrame({"open": self.bid, "high": self.bid, "low": self.bid,
                                          "close": self.bid,
                                          "dividends": np.zeros_like(self.dividends)}, index=self.dates)
        self.dict_df = {self.symbol_list[0]: self.df}
        self.queue = Queue()
        self.data = GeneratedHistoricalData(self.queue, self.symbol_list, self.dict_df,
                                            bid_offer={s: self.bid_offer for s in self.symbol_list})
        self.data_no_dividends = GeneratedHistoricalData(self.queue, self.symbol_list,
                                                         {self.symbol_list[0]: df_no_divividends},
                                                         bid_offer={s: self.bid_offer for s in self.symbol_list})
        self.instrument_dict = {"CfD": partial(CfDInstrument, rates=self.data.rates),
                                "Stock": StockInstrument}
        pass

    def test_trade_cost(self):
        """Test that closing a deal returns the correct cost (bid offer for CFD, full cash for stocks)"""
        cfd = CfDInstrument(rates=self.data.rates)
        stock = StockInstrument()
        stock_tobin = StockTobinInstrument()
        t = 0
        for volume in 1, 10, 100:
            bid = self.bid[t]
            ask = self.ask[t]
            date = self.dates[t]
            # Opens CfD position
            cdf_buy_cash = cfd.trade(self.equity, volume, bid=bid, offer=ask, date=date)
            # Closes position (this won't have any bid_offer cost)
            cfd_sell_cash = cfd.trade(self.equity, -volume, bid=bid, offer=ask, date=date)
            # Reopens position again (here bid_offer cost will be included)
            cfd_sell_cash = cfd.trade(self.equity, -volume, bid=bid, offer=ask, date=date)
            stock_buy_cash = stock.trade(self.equity, volume, bid=bid, offer=ask, date=date)
            stock_sell_cash = stock.trade(self.equity, -volume, bid=bid, offer=ask, date=date)
            stock_tobin_buy_cash = stock_tobin.trade(self.equity, volume, bid=bid, offer=ask, date=date)
            self.assertEqual(cdf_buy_cash.cash, cfd_sell_cash.cash / 2, "Buy and sell CfDs don't return correct prices")
            self.assertEqual(stock_sell_cash.cash, -bid * volume, "Incorrect sell cost")
            self.assertEqual(stock_buy_cash.cash, ask * volume, "Incorrect buy cost")
            self.assertEqual(stock_tobin_buy_cash.others, ask * volume * 0.002, "Incorrect buy tax cost")
            if volume == 1:
                print(f"{stock_sell_cash=} {stock_buy_cash=} {cfd_sell_cash=} {cdf_buy_cash=}")

    def test_mtm_bars(self):
        """Tests mar-to-market calculations using bars"""
        for volume in 1, -1:
            for s in self.symbol_list:
                for instrument_name, instrument in self.instrument_dict.items():
                    self.data.reset()
                    self.data.update_bars()
                    inst = instrument()
                    bar = self.data.get_latest_bars(s)[-1]
                    close_bid = bar.close_bid
                    close_ask = bar.close_ask
                    if volume < 0 and isinstance(inst, StockInstrument):
                        with self.assertRaises(PositionBelowMinimumException):
                            inst.trade(self.equity, volume, close_bid, close_ask, bar.timestamp)
                        inst = None
                    else:
                        inst.trade(self.equity, volume, close_bid, close_ask, bar.timestamp)
                    while self.data.continue_backtest:
                        self.data.update_bars()
                        bar = self.data.get_latest_bars(s)[-1]
                        market_bid = bar.close_bid
                        market_ask = bar.close_ask
                        if inst is not None:
                            inst.new_day(bar)
                            mtm = inst.mtm(market_bid, market_ask)
                            print(f"{instrument_name}: {volume=} {close_bid=} {close_ask=} {market_bid=} {market_ask=} "
                                  f"{mtm=}")
                            print(f"{instrument_name}: {inst.total_commissions=}")

    def test_event_data_signals(self):
        """Makes sure that signals' queue works properly"""
        while True:
            if self.data.continue_backtest:
                self.data.update_bars()
            else:
                break
            event = self.queue.get()
            print(event)

    def test_dividends(self):
        """Test dividends calculations, specially for CfD. For CfD dividends must have NO impact"""
        for instrument_name, instrument_obj in self.instrument_dict.items():
            for position in 1, -1:
                instrument = instrument_obj()
                if isinstance(instrument, StockInstrument) and position < 0:
                    continue    # Avoids exception caused by short stock positions
                while self.data.continue_backtest:
                    self.data.update_bars()
                    bar = self.data.get_latest_bars(self.symbol_list[0])[0]
                    print(bar)
                    instrument.trade(self.equity, position - instrument.position,
                                     bar.close_bid, bar.close_ask, bar.timestamp)
                    mtm = instrument.mtm(bar.close_bid, bar.close_ask)
                    print(f"Instrument before new_day: {instrument}")
                    self.data.update_bars()
                    next_bar = self.data.get_latest_bars(self.symbol_list[0])[0]
                    dividends = next_bar.dividends
                    instrument.new_day(bar, dividends)
                    mtm_new_day = instrument.mtm(bar.close_bid, bar.close_ask)
                    if isinstance(instrument, StockInstrument):
                        self.assertEqual(position * dividends, mtm_new_day - mtm,
                                         f"Dividends are not incorporated to Stocks for day {bar.timestamp}")
                    elif isinstance(instrument, CfDInstrument):
                        self.assertEqual(mtm, mtm_new_day,
                                         f"CfD dividends are not correctly calculated for day {bar.timestamp}")
                    else:
                        raise NotImplementedError("Unknown instrument: test not implemented")

                    print(f"Instrument After new_day: {instrument}")
                self.data.reset()

    def test_close(self):
        """Test close and reopen of positions, both in CdD and stocks. Test mtm and commissions"""
        expected_positions_cases = {
            "Case 1": [1, 1, -1, -1, 1, 1],
            "Case 2x": [2, 2, -2, -2, 2, 2],  # Same as before, but doubled
            "Case delta": [1, 0.5, 0, -.5, -1, -1],  # including a 0.5 intermediate
        }
        for case, expected_position_case in expected_positions_cases.items():
            for instrument_name, instrument in self.instrument_dict.items():
                inst = instrument()
                t = 0
                expected_position_list = list()
                closings = list()
                bids = list()
                asks = list()
                mtms = list()
                expected_positions_copy = expected_position_case.copy()
                while self.data.continue_backtest:
                    expected_position = expected_positions_copy.pop()
                    self.data.update_bars()
                    last_bar = self.data.get_latest_bars(self.symbol_list[0])[-1]
                    if isinstance(inst, StockInstrument):
                        expected_position = max(0, expected_position)
                    expected_position_list.append(expected_position)
                    closings.append(last_bar.close)
                    bids.append(last_bar.close_bid)
                    asks.append(last_bar.close_ask)
                    inst.trade(self.equity, expected_position - inst.position, last_bar.close_bid, last_bar.close_ask,
                               last_bar.timestamp)
                    mtm = inst.mtm(last_bar.close_bid, last_bar.close_ask)
                    print(
                        f"{t=} {case=} {instrument_name}: {expected_position=} {inst.position=} {mtm=} {last_bar.close_bid=}"
                        f" {last_bar.close_ask=} {inst.trade_price=} {inst.total_commissions=} {inst.total_mtm=}")
                    inst.new_day(last_bar)
                    print(f"{t=} {case=} {instrument_name}: {expected_position=} {inst.total_mtm=}")
                    mtms.append(inst.total_mtm)
                    t = t + 1

                # Calculate mtm
                np_prices = np.array(closings)
                np_positions = np.array(expected_position_list)
                # calculate vectorized_pnl
                vectorized_pnl = pnl_positions(np_positions, np.array(bids), np.array(asks))

                # Without bid/offer
                calc_mtms_without_bidoffer = np.zeros_like(np_positions)
                calc_mtms_without_bidoffer[1:] = np.cumsum(np_positions[:-1] * np.diff(np_prices))
                # Calculate entry points
                entry_points = calculate_entry_points(np_positions)
                bid_offer_value = -np.cumsum(entry_points * self.bid_offer * 2)
                # Including effect of bid/offer
                calc_mtms = calc_mtms_without_bidoffer + bid_offer_value
                self.assertSequenceEqual(mtms, calc_mtms.tolist(), f"Bad pnl found for instrument '{instrument_name}' "
                                                                   f"in case '{case}'. "
                                                                   f"{entry_points=} {calc_mtms_without_bidoffer=}")
                self.assertSequenceEqual(vectorized_pnl.tolist(), calc_mtms.tolist(),
                                                                   f"Bad vectorized pnl found for instrument '{instrument_name}' "
                                                                   f"in case '{case}'. "
                                                                   f"{entry_points=} {calc_mtms_without_bidoffer=}")
                self.data.reset()

    def test_vectorized_pnl(self):
        """Tests calculation of vectorized pnl vs rule-based one: time to calculate and same result"""
        n = 3000
        iterations = 1000
        positions = np.random.rand(n) * 2 - 1
        for function in (calculate_entry_points, calculate_entry_points_vectorized):
            tic = time()
            for _ in range(iterations):
                function(positions)
            res = time() - tic
            print(f"Elapsed time for {function.__name__}: {res:.3f}s")
        self.assertTrue(np.all(calculate_entry_points(positions) == calculate_entry_points_vectorized(positions)))

    def test_position_change(self):

        cases = [
            {"old_pos": 0, "new_pos": 1, "additional_pos": 1, "closing_pos": 0},
            {"old_pos": 1, "new_pos": 1, "additional_pos": 0, "closing_pos": 0},
            {"old_pos": .5, "new_pos": 1, "additional_pos": 0.5, "closing_pos": 0},
            {"old_pos": .5, "new_pos": 0, "additional_pos": 0, "closing_pos": -0.5},
            {"old_pos": .5, "new_pos": -1, "additional_pos": -1, "closing_pos": -0.5},
            {"old_pos": 0, "new_pos": -1, "additional_pos": -1, "closing_pos": 0},
            {"old_pos": -1, "new_pos": -1, "additional_pos": 0, "closing_pos": 0},
            {"old_pos": -1, "new_pos": -1.5, "additional_pos": -0.5, "closing_pos": 0},
            {"old_pos": -1, "new_pos": -.5, "additional_pos": 0, "closing_pos": 0.5},
            {"old_pos": -1, "new_pos": .5, "additional_pos": 0.5, "closing_pos": 1},
        ]
        for case in cases:
            instrument = StockInstrument()
            instrument.position = case['old_pos']
            trade_volume = case['new_pos'] - case['old_pos']
            calc_add, calc_close = instrument.calculate_new_positions(trade_volume)
            details = f"{case=} {calc_add=} {calc_close=}"
            self.assertEqual(case['additional_pos'], calc_add, f"Bad additional position: {details}")
            self.assertEqual(case['closing_pos'], calc_close, f"Bad closing position: {details}")
        pass


if __name__ == '__main__':
    unittest.main()
