"""
Vectorized version of event_driven for fast prototyping
"""
import numpy as np


def calculate_entry_points_vectorized(positions):
    """
    Calculates entry points (incremental positions that are not closing positions so bid-offer must be applied to them)
    This is a vectorized version that uses numpy to speed up calculations
    :param positions: positions for each point (cumsum of orders)
    :return: the positions that are incremental for each point to apply bid-offer cost to it
    """
    inc_positions = np.diff(positions, prepend=0)
    abs_positions = np.abs(positions)
    inc_abs_positions = np.diff(abs_positions, prepend=0)
    retval = np.zeros_like(positions)
    # Treatment of incremental positions: Positions only increment where is the same sign
    retval += np.where(np.sign(inc_positions) == np.sign(positions), inc_abs_positions, 0)
    # adjust sign change
    sign_change = np.sign(positions[:-1]) * np.sign(positions[1:]) == -1
    retval[1:][sign_change] = abs_positions[1:][sign_change]
    return retval


def calculate_entry_points(positions):
    """
    Calculates entry points (incremental positions that are not closing positions so bid-offer must be applied to them)
    This is a non vectorized version that is slower but easier to debug
    :param positions: positions for each point (cumsum of orders)
    :return: the positions that are incremental for each point to apply bid-offer cost to it
    """
    entry_points = list()
    for i in range(len(positions)):
        entry_points.append(0)
        if i == 0 and positions[0] != 0:
            entry_points[i] = abs(positions[0])
        # If close position
        elif np.sign(positions[i - 1]) != np.sign(positions[i]) and positions[i] != 0:
            entry_points[i] = abs(positions[i])
        # if increase position
        elif abs(positions[i]) > abs(positions[i - 1]):
            entry_points[i] = abs(positions[i]) - abs(positions[i - 1])
    entry_points = np.array(entry_points)
    return entry_points


def pnl_positions(positions: np.ndarray, bid: np.ndarray, offer: np.ndarray):
    """Pnl for positions. It just differentiates it and calculates pnl using orders. See pnl_ordes"""
    bid_offer = offer - bid
    inc_bid = np.diff(bid)
    inc_offer = np.diff(offer)  # , prepend=offer[0])
    # First: compute pnl just using bid or offer
    calc_mtms_without_bidoffer = np.zeros_like(positions)
    calc_mtms_without_bidoffer[1:] = np.cumsum(positions[:-1] * np.where(positions[:-1] > 0, inc_bid, inc_offer))

    # Now: look for bid_offer.
    # bid-offer must be charged only when positions increases
    # Calculate entry points

    bid_offer_cost = -np.cumsum(calculate_entry_points(positions) * bid_offer)

    calc_pnl = calc_mtms_without_bidoffer + bid_offer_cost
    return calc_pnl


def pnl_orders(orders: np.ndarray, bid: np.ndarray, offer: np.ndarray):
    """
    Computes total pnl for the given orders
    :param orders: the orders, + for long (buy) and - for short (sell)
    :param bid: bid prices. Long positions are valued against them position
    :param offer: offer/ask prices. Short positions are valued against them
    :return: the cumulative pnl including bid_offer
    """
    return pnl_orders(np.cumsum(orders), bid, offer)


if __name__ == '__main__':
    import pandas as pd


    def print_summary(**kwargs):
        df = pd.DataFrame(kwargs)
        # df['positions'] = positions
        # df['bid'] = bid
        # df['offer'] = offer
        # df['pnl'] = pnl
        print(df)


    positions = np.array([1, 2, 1, 0, -1, -2, -1, 1, -1])
    print_summary(positions=positions, entry=calculate_entry_points(positions),
                  vectorized=calculate_entry_points_vectorized(positions))
    # exit(0)
    bid = 40 + positions
    offer = bid + 0.5

    pnl = pnl_positions(positions, bid, bid)
    print_summary(positions=positions, bid=bid, offer=offer, pnl=pnl)

    pnl = pnl_positions(positions, bid, offer)
    print_summary(positions=positions, bid=bid, offer=offer, pnl=pnl)
