"""
The MIT License (MIT)

Copyright (c) 2016 Tito Ingargiola
Copyright (c) 2019 Stefan Jansen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from __future__ import annotations

from typing import Type

import gym
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

from ong_trading.event_driven.data import YahooHistoricalData
from ong_trading.features.preprocess import MLPreprocessor
from ong_trading import logger as log
from ong_trading.ML.RL.config import ModelConfig
from ong_trading.utils.utils import fmt_dt


class DataSource:
    """
    Data source for TradingEnvironment

    Loads & preprocesses daily price & volume data
    Provides data for each new episode.
    Stocks with longest history:

    ticker  # obs
    KO      14155
    GE      14155
    BA      14155
    CAT     14155
    DIS     14155

    """

    def __init__(self, model_name: str, trading_days=252, ticker='AAPL',
                 train_start: pd.Timestamp = None,
                 validation_start: pd.Timestamp = None,
                 test_start: pd.Timestamp = None,
                 preprocessor: Type[MLPreprocessor] = None):
        """
        Inits data and splits it into train, validation and test data
        :param model_name: name of the model (needed for storing preprocessors)
        :param trading_days: number of trading days (for minimum dates in step)
        :param ticker: name of the ticker to read
        :param train_start: start date for train. If None then data from the fist date available data will be used
        :param validation_start: start date for validation. If None, then test_start_date will be used
        :param test_start: start date for test.
        :param preprocessor: a preprocessor class for preprocess data (on train data only)
        """
        self.model_name = model_name
        self.ticker = ticker
        self.trading_days = trading_days
        self.data_source = YahooHistoricalData(None, [self.ticker])
        self.preprocessor = preprocessor()

        validation_start = validation_start or test_start
        all_data = self.load_data()
        if train_start:
            all_data = all_data[train_start:].dropna()  # Just in case
        all_index = all_data.index

        self._data_types = ("train", "validation", "test")
        self._features = dict()
        self._data = dict()
        self._returns = dict()

        # Do splits between train, validation and test
        for idx, (type_, start_data, end_data) in enumerate(zip(
                self._data_types,
                (train_start, validation_start, test_start),
                (validation_start, test_start, None))):
            self._data[type_] = all_data[start_data:end_data].dropna()  # Just in case
            if type_ == "train":
                self.preprocessor.fit(self._data[type_])
                self.preprocessor.save(ModelConfig.model_path("preprocessor"))
            self._features[type_] = self.preprocessor.transform(all_data)[start_data:end_data].dropna()
            self._returns[type_] = self._data[type_].adj_close.pct_change()
            type_index = self._data[type_].index
            if type_index.empty:
                log.info(f"No data for {type_} is used")
            else:
                log.info(f"{type_} data from: {len(type_index)} data "
                         f"({len(type_index) / len(all_index):.2%} of all data) "
                         f"from {fmt_dt(type_index[0])} to {fmt_dt(type_index[-1])}")

        # Assert any data is not overlapping
        for idx in range(1, len(self._data_types)):
            this_type = self._data_types[idx]
            if not self._features[this_type].index.empty:
                for prev_type in self._data_types[:(idx - 1)]:
                    if not self._features[prev_type].index.empty:
                        this_start = self._features[this_type].index[0]
                        prev_end = self._features[prev_type].index[-1]
                        assert this_start > prev_end, f"Indexes overlap for types {this_type}({fmt_dt(this_start)}) " \
                                                      f"and {prev_type}({fmt_dt(prev_end)})"


        # # Split between train and test (out of sample) data using train_split_data
        # # Gives the index of the PREVIOUS data (if date not found)
        # index = all_data.index
        # # train_data_slice = slice(None, train_test_split_date)
        # # test_data_slice = slice(train_data_slice, None)
        # self.train_orig_data = all_data[:train_test_split_date]
        # log.info(f"Using {len(self.train_orig_data.index)} training data up to {self.train_orig_data.index[-1]}")
        # assert self.train_orig_data.index[-1] < pd.Timestamp(train_test_split_date), \
        #     "Training data includes the first validation data"
        # self.test_orig_data = all_data[train_test_split_date:]
        # log.info(f"Using {len(self.test_orig_data.index)} "
        #          f"({len(self.test_orig_data.index) / len(all_data.index):.2%} of all data) "
        #          f"test data up to {self.test_orig_data.index[-1]}")
        # assert self.test_orig_data.index[0] >= pd.Timestamp(train_test_split_date), \
        #     "Test data includes the last train data"
        # # Train data
        # self.train_data = self.preprocessor.transform(self.train_orig_data)
        # self.train_returns = self.train_orig_data.adj_close.pct_change()
        # self.preprocessor.save(ModelConfig.model_path("preprocessor"))
        # # Test data
        # self.test_data = self.preprocessor.transform(all_data)[train_test_split_date:]
        # self.test_returns = all_data.adj_close[train_test_split_date:].pct_change()
        self.min_values = self.train_features.min().to_numpy()
        self.max_values = self.train_features.max().to_numpy()
        self.step = 0
        self.offset = None

    @property
    def data_types(self) -> tuple:
        return self._data_types

    @property
    def train_features(self) -> pd.DataFrame:
        return self._features['train']

    @property
    def validation_features(self) -> pd.DataFrame:
        return self._features['validation']

    @property
    def test_features(self) -> pd.DataFrame:
        return self._features['test']

    def features(self, type_: str) -> pd.DataFrame:
        return self._features.get(type_)

    @property
    def train_data(self) -> pd.DataFrame:
        return self._data['train']

    @property
    def validation_data(self) -> pd.DataFrame:
        return self._data['validation']

    @property
    def test_data(self) -> pd.DataFrame:
        return self._data['test']

    def data(self, type_: str) -> pd.DataFrame:
        return self._data.get(type_)

    @property
    def train_returns(self) -> pd.DataFrame:
        return self._returns['train']

    @property
    def validation_returns(self) -> pd.DataFrame:
        return self._returns['validation']

    @property
    def test_returns(self) -> pd.DataFrame:
        return self._returns['test']

    def returns(self, type_: str) -> pd.DataFrame:
        return self._returns.get(type_)

    def load_data(self):
        log.info('loading data for {}...'.format(self.ticker))
        df = self.data_source.to_pandas(self.ticker)
        # idx = pd.IndexSlice
        # with pd.HDFStore('../data/assets.h5') as store:
        #     df = (store['quandl/wiki/prices']
        #           .loc[idx[:, self.ticker],
        #                ['adj_close', 'adj_volume', 'adj_low', 'adj_high']]
        #           .dropna()
        #           .sort_index())
        # df = df.loc[:, ['adj_close', 'adj_volume', 'adj_low', 'adj_high']]
        df = df.loc[:, ['adj_close', 'close', 'volume', 'low', 'high', 'open']]
        df = df.dropna()  # Just in case...
        # df.columns = ['close', 'volume', 'low', 'high']

        log.info('got data for {}...'.format(self.ticker))
        return df

    def reset(self):
        """Provides starting index for time series and resets step"""
        high = len(self.train_features.index) - self.trading_days
        self.offset = np.random.randint(low=0, high=high)
        self.step = 0

    def take_step(self):
        """Returns data for current trading day and done signal"""
        obs = self.train_features.iloc[self.offset + self.step].values
        if self.offset + self.step == 218:
            pass
        ret = self.train_returns.iat[self.offset + self.step]
        self.step += 1
        done = self.step > self.trading_days
        return (ret, obs), done

    def take_all_steps(self):
        """Returns data for all steps at once"""
        obs = self.train_features.iloc[self.offset:(self.offset + self.trading_days)].values
        rets = self.train_returns.iloc[self.offset:(self.offset + self.trading_days)].values
        return rets, obs


class TradingSimulator:
    """ Implements core trading simulator for single-instrument univ """

    def __init__(self, steps, trading_cost_bps, time_cost_bps):
        # invariant for object life
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.steps = steps

        # change every step
        self.step = 0
        self.actions = np.zeros(self.steps)
        self.navs = np.ones(self.steps)
        self.market_navs = np.ones(self.steps)
        self.strategy_returns = np.ones(self.steps)
        self.positions = np.zeros(self.steps)
        self.costs = np.zeros(self.steps)
        self.trades = np.zeros(self.steps)
        self.market_returns = np.zeros(self.steps)

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.navs.fill(1)
        self.market_navs.fill(1)
        self.strategy_returns.fill(0)
        self.positions.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.market_returns.fill(0)

    def take_step(self, action, market_return):
        """ Calculates NAVs, trading costs and reward
            based on an action and latest market return
            and returns the reward and a summary of the day's activity. """

        start_position = self.positions[max(0, self.step - 1)]
        start_nav = self.navs[max(0, self.step - 1)]
        start_market_nav = self.market_navs[max(0, self.step - 1)]
        self.market_returns[self.step] = market_return
        self.actions[self.step] = action

        end_position = action - 1  # short, neutral, long
        n_trades = end_position - start_position
        self.positions[self.step] = end_position
        self.trades[self.step] = n_trades

        trade_costs = abs(n_trades) * self.trading_cost_bps
        time_cost = 0 if n_trades else self.time_cost_bps
        self.costs[self.step] = trade_costs + time_cost
        # reward = start_position * market_return - self.costs[self.step]
        reward = start_position * market_return - self.costs[max(0, self.step - 1)]
        self.strategy_returns[self.step] = reward

        if self.step != 0:
            self.navs[self.step] = start_nav * (1 + self.strategy_returns[self.step])
            self.market_navs[self.step] = start_market_nav * (1 + self.market_returns[self.step])

        info = {'reward': reward,
                'nav': self.navs[self.step],
                'costs': self.costs[self.step]}

        self.step += 1
        return reward, info

    def result(self):
        """returns current state as pd.DataFrame """
        return pd.DataFrame({'action': self.actions,  # current action
                             'nav': self.navs,  # starting Net Asset Value (NAV)
                             'market_nav': self.market_navs,
                             'market_return': self.market_returns,
                             'strategy_return': self.strategy_returns,
                             'position': self.positions,  # eod position
                             'cost': self.costs,  # eod costs
                             'trade': self.trades})  # eod trade)


class TradingEnvironment(gym.Env):
    """A simple trading environment for reinforcement learning.

    Provides daily observations for a stock price series
    An episode is defined as a sequence of 252 trading days with random start
    Each day is a 'step' that allows the agent to choose one of three actions:
    - 0: SHORT
    - 1: HOLD
    - 2: LONG

    Trading has an optional cost (default: 10bps) of the change in position value.
    Going from short to long implies two trades.
    Not trading also incurs a default time cost of 1bps per step.

    An episode begins with a starting Net Asset Value (NAV) of 1 unit of cash.
    If the NAV drops to 0, the episode ends with a loss.
    If the NAV hits 2.0, the agent wins.

    The trading simulator tracks a buy-and-hold strategy as benchmark.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 trading_days=252,
                 trading_cost_bps=1e-3,
                 time_cost_bps=1e-4,
                 ticker='AAPL',
                 model_name: str = "",
                 train_start: pd.Timestamp | int = None,
                 validation_start: pd.Timestamp | int = None,
                 test_start: pd.Timestamp | int = None,
                 preprocessor=None,
                 ):
        self.model_name = model_name
        self.trading_days = trading_days
        self.trading_cost_bps = trading_cost_bps
        self.ticker = ticker
        self.time_cost_bps = time_cost_bps
        self.data_source = DataSource(trading_days=self.trading_days,
                                      ticker=ticker, model_name=self.model_name,
                                      train_start=train_start,
                                      validation_start=validation_start,
                                      test_start=test_start,
                                      preprocessor=preprocessor)
        self.simulator = TradingSimulator(steps=self.trading_days,
                                          trading_cost_bps=self.trading_cost_bps,
                                          time_cost_bps=self.time_cost_bps)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.data_source.min_values,
                                            self.data_source.max_values, dtype=np.float64)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """Returns state observation, reward, done and info"""
        assert self.action_space.contains(action), '{} {} invalid'.format(action, type(action))
        (market_return, observation), done = self.data_source.take_step()
        reward, info = self.simulator.take_step(action=action,
                                                # market_return=observation[0])
                                                market_return=market_return)
        # return observation, reward, done, info
        terminated = done
        truncated = False
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Resets DataSource and TradingSimulator; returns first observation"""
        self.data_source.reset()
        self.simulator.reset()
        return self.data_source.take_step()[0][1], dict()
        # return self.data_source.take_step()[0]

    # TODO
    def render(self, mode='human'):
        """Not implemented"""
        pass
