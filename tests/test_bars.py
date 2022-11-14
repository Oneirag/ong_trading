import unittest
from queue import Queue
from ong_utils import OngTimer
from ong_trading.event_driven.data import YahooHistoricalData


class TestData(unittest.TestCase):
    """Simple tests for data"""

    def setUp(self) -> None:
        self.events = Queue()

    def test_reset_bid_offer(self):
        """Tests that bars indeed are reset with reset() method and also that bid-offer works"""

        bid_offer = 1
        symbols = ["ELE.MC", "SAN.MC"]
        with OngTimer(msg="create bars", enabled=False):
            bars = YahooHistoricalData(events=self.events, symbol_list=symbols,
                                       bid_offer={"ELE.MC": bid_offer})
        n_bars = list()
        for rep in range(5):
            while bars.continue_backtest:
                bars.update_bars()
                data = bars.get_latest_bars(symbol=symbols[0])[-1]
                ###################
                # Bid offer check
                ###################
                for quote in "open", "high", "low", "close":
                    # This is aprox 3x faster than calling getattr
                    # self.assertEqual(data.open_ask - data.open, bid_offer)
                    # self.assertEqual(data.open_bid - data.open, -bid_offer)
                    for side, spread in {"bid": -bid_offer, "ask": bid_offer}.items():
                        self.assertAlmostEqual(getattr(data, quote + "_" + side) - getattr(data, quote), spread,
                                               msg=f"Wrong bid offer spread for {quote}_{side} in {data}")
            n_bars.append(self.events.qsize())
            print(f"Rep {rep}, n_bars={n_bars[-1]}")
            bars.reset()
            # Empty queue
            self.events.queue.clear()

            self.assertSequenceEqual(n_bars, [n_bars[0]] * len(n_bars),
                                     "Bars not generated equally")
            # all bars updated, now restart


if __name__ == '__main__':
    unittest.main()
