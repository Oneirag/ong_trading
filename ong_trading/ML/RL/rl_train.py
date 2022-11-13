from pathlib import Path
from time import time
from dataclasses import asdict

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import tensorflow as tf

import gym
from gym.envs.registration import register

from ong_trading.ML.RL.config import ModelHyperParams, ModelConfig
from ong_trading.ML.RL.ddqna_agent import DDQNAgent
from ong_trading.event_driven.performance import annualised_returns
from ong_trading.utils.utils import format_time
from ong_trading.vectorized.pnl import pnl_positions
from ong_trading.ML.RL.trading_env import DataSource
from ong_trading.event_driven.performance import OutputAnalyzer

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

np.random.seed(ModelConfig.random_seed)
tf.random.set_seed(ModelConfig.random_seed)

# Check if using GPU
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print('Using CPU')


class Evaluator:
    """Gives the annualised valuation of the strategy in the whole training and validation data set"""

    strategy_names = "benchmark", "strategy"

    def __init__(self, ddqn: DDQNAgent, data_source: DataSource):
        self.ddqn = ddqn
        self.data_source = data_source
        self.strat = dict()
        # Stored for testing purposes
        self.features = dict()
        self.positions = dict()
        self.prices = dict()
        self.initial_cash = dict()
        self.pnl = dict()

        for data_type in self.data_source.data_types:
            features = self.data_source.features(data_type)
            if features.empty:
                continue
            else:
                self.features[data_type] = features.values
                self.pnl[data_type] = dict()
                self.strat[data_type] = dict()
            qs = self.ddqn.online_network.predict_on_batch(self.features[data_type])
            self.positions[data_type] = np.argmax(qs, axis=1).astype(float) - 1.0
            data = data_source.data(data_type)
            self.prices[data_type] = data.adj_close.values[-len(self.positions[data_type]):]
            self.initial_cash[data_type] = data.close.values[-len(self.positions[data_type]):][0]
            for strategy_name in self.strategy_names:
                if strategy_name == "benchmark":
                    positions = np.ones_like(self.positions[data_type])
                else:
                    positions = self.positions[data_type]
                self.pnl[data_type][strategy_name] = pnl_positions(positions,
                                                                   bid=self.prices[data_type],
                                                                   offer=self.prices[data_type])
                self.strat[data_type][strategy_name] = OutputAnalyzer.from_pnl(self.pnl[data_type][strategy_name],
                                                                               initial_cash=self.initial_cash[data_type])

    def __str__(self):
        """ Returns a string with representation of results for all available data_types"""
        text = ""
        for data_type in self.data_source.data_types:
            if data_type in self.strat:
                strat_oa = self.strat[data_type]['strategy']
                bench_oa = self.strat[data_type]['benchmark']
                if strat_oa is not None:
                    text += f"{data_type}: {strat_oa.total_return_pct_annualised(): >7.2%} " \
                            f"({bench_oa.total_return_pct_annualised(): >7.2%} mkt) " \
                            f"Sh: {strat_oa.sharpe(): >7.2%}| "
        return text


def create_tradingenv_ddqn() -> tuple:
    # Set up environment
    register(
        id='trading-v0',
        entry_point='ong_trading.ML.RL.trading_env:TradingEnvironment',
        max_episode_steps=ModelConfig.trading_days,
        kwargs=dict(ticker=ModelConfig.ticker,
                    model_name=ModelConfig.full_model_name,
                    trading_cost_bps=ModelConfig.trading_cost_bps,
                    time_cost_bps=ModelConfig.time_cost_bps,
                    train_start=ModelConfig.train_start,
                    validation_start=ModelConfig.validation_start,
                    test_start=ModelConfig.test_start,
                    preprocessor=ModelConfig.preprocessor)
    )

    # Initialize trading environment
    trading_environment = gym.make('trading-v0')
    trading_environment.seed(ModelConfig.random_seed)

    # Get environ params
    state_dim = trading_environment.observation_space.shape[0]
    num_actions = trading_environment.action_space.n

    # Create DDQN Agent
    # We will use TensorFlow to create our Double Deep Q-Network .
    tf.keras.backend.clear_session()

    ddqn = DDQNAgent(state_dim=state_dim, num_actions=num_actions, model_name=ModelConfig.full_model_name,
                     random_seed=ModelConfig.random_seed,
                     **asdict(ModelHyperParams()))

    ddqn.online_network.summary()
    return trading_environment, ddqn


# Visualization
def track_results(episode, nav_ma_100, nav_ma_10,
                  market_nav_100, market_nav_10,
                  win_ratio, total, epsilon, extra_text):
    template = '{:>4d} | {} | Agent: {:>6.1%} ({:>6.1%}) | '
    template += 'Market: {:>6.1%} ({:>6.1%}) | '
    template += 'Wins: {:>5.1%} | eps: {:>6.3f}'
    template += f" | {extra_text}"
    print(template.format(episode, format_time(total),
                          nav_ma_100 - 1, nav_ma_10 - 1,
                          market_nav_100 - 1, market_nav_10 - 1,
                          win_ratio, epsilon))


def train(max_episodes: int = ModelConfig.max_episodes) -> dict:

    trading_environment, ddqn = create_tradingenv_ddqn()
    max_episode_steps = trading_environment.spec.max_episode_steps

    # Initialize variables
    navs, market_navs, diffs = [], [], []
    start = time()
    results = []
    for episode in range(1, max_episodes + 1):
        # (this_return, this_state), info = trading_environment.reset()
        trading_environment.reset()

        # Calculate all at once for faster execution (steps are independent of actions as agent cannot alter market)
        _, all_states = trading_environment.data_source.take_all_steps()
        all_actions = iter(ddqn.epsilon_greedy_policy_all(all_states))
        this_state = all_states[0]

        for episode_step in range(max_episode_steps):
            # action = ddqn.epsilon_greedy_policy(this_state.reshape(-1, state_dim))
            action = next(all_actions)
            next_state, reward, done, *_ = trading_environment.step(action)

            ddqn.memorize_transition(this_state,
                                     action,
                                     reward,
                                     next_state,
                                     0.0 if done else 1.0)
            if ddqn.train:
                ddqn.experience_replay()
            if done:
                break
            this_state = next_state

        result = trading_environment.env.simulator.result()
        final = result.iloc[-1]

        nav = final.nav * (1 + final.strategy_return)
        navs.append(nav)

        market_nav = final.market_nav
        market_navs.append(market_nav)

        diff = nav - market_nav
        diffs.append(diff)
        if episode % 10 == 0:
            ddqn.save_model()
            track_results(episode, np.mean(navs[-100:]), np.mean(navs[-10:]),
                          np.mean(market_navs[-100:]), np.mean(market_navs[-10:]),
                          np.sum([s > 0 for s in diffs[-100:]]) / min(len(diffs), 100),
                          time() - start, ddqn.epsilon,
                          extra_text=str(Evaluator(ddqn, trading_environment.data_source)))
        # if len(diffs) > 25 and all([r > 0 for r in diffs[-25:]]):
        #     print(result.tail())
        #     break

    ddqn.print_timers()
    ddqn.save_model()
    trading_environment.close()

    # Store Results
    results = pd.DataFrame({'Episode': list(range(1, episode + 1)),
                            'Agent': navs,
                            'Market': market_navs,
                            'Difference': diffs}).set_index('Episode')

    results['Strategy Wins (%)'] = (results.Difference > 0).rolling(100).sum()
    results.info()
    return results


def plot_results(training_results: pd.DataFrame):
    # make dirs for results
    results_path = Path('results', 'trading_bot')
    if not results_path.exists():
        results_path.mkdir(parents=True)

    with sns.axes_style('white'):
        # sns.distplot(results.Difference)
        sns.histplot(training_results.Difference)
        sns.despine()

    fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharey=True)

    df1 = (training_results[['Agent', 'Market']]
           .sub(1)
           .rolling(100)
           .mean())
    df1.plot(ax=axes[0],
             title='Annual Returns (Moving Average)',
             lw=1)

    df2 = training_results['Strategy Wins (%)'].div(100).rolling(50).mean()
    df2.plot(ax=axes[1],
             title='Agent Outperformance (%, Moving Average)')

    for ax in axes:
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: '{:,.0f}'.format(x)))
    axes[1].axhline(.5, ls='--', c='k', lw=1)

    sns.despine()
    fig.tight_layout()
    fig.savefig(results_path / 'performance', dpi=300)


if __name__ == '__main__':

    results = train(ModelConfig.max_episodes)
    plot_results(results)


    """
    10 | 00:00:00 | Agent: -20.3% (-20.3%) | Market:   6.1% (  6.1%) | Wins: 20.0% | eps:  0.960
2022-10-25 06:43:15.805938: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
  20 | 00:00:26 | Agent: -20.4% (-20.4%) | Market:   1.5% ( -3.2%) | Wins: 25.0% | eps:  0.921
  30 | 00:01:43 | Agent: -20.7% (-21.5%) | Market:   5.0% ( 12.0%) | Wins: 20.0% | eps:  0.881
  40 | 02:46:30 | Agent: -19.2% (-14.4%) | Market:   7.8% ( 16.2%) | Wins: 22.5% | eps:  0.842
  50 | 05:34:03 | Agent: -19.4% (-20.4%) | Market:   9.0% ( 13.7%) | Wins: 20.0% | eps:  0.802
  60 | 08:34:54 | Agent: -21.2% (-30.4%) | Market:   8.4% (  5.4%) | Wins: 18.3% | eps:  0.762
  70 | 11:59:08 | Agent: -20.6% (-16.5%) | Market:   9.1% ( 13.5%) | Wins: 18.6% | eps:  0.723
  80 | 14:59:10 | Agent: -20.1% (-16.5%) | Market:  11.0% ( 24.2%) | Wins: 16.2% | eps:  0.683
  90 | 15:00:22 | Agent: -20.6% (-24.6%) | Market:  10.6% (  7.4%) | Wins: 16.7% | eps:  0.644
 100 | 15:01:31 | Agent: -20.7% (-22.1%) | Market:  10.7% ( 12.0%) | Wins: 16.0% | eps:  0.604
 110 | 15:02:43 | Agent: -20.9% (-22.2%) | Market:  11.0% (  8.8%) | Wins: 14.0% | eps:  0.564
 120 | 15:03:55 | Agent: -20.7% (-18.5%) | Market:  13.7% ( 23.7%) | Wins: 12.0% | eps:  0.525
 130 | 15:05:08 | Agent: -20.6% (-20.1%) | Market:  13.2% (  7.2%) | Wins: 14.0% | eps:  0.485
 140 | 15:06:19 | Agent: -20.7% (-15.5%) | Market:  12.9% ( 13.0%) | Wins: 14.0% | eps:  0.446
 150 | 15:07:34 | Agent: -19.7% (-11.0%) | Market:  12.3% (  7.2%) | Wins: 17.0% | eps:  0.406
 160 | 15:08:46 | Agent: -18.6% (-19.1%) | Market:  11.2% ( -5.3%) | Wins: 19.0% | eps:  0.366
 170 | 15:09:59 | Agent: -19.5% (-25.0%) | Market:  10.8% (  9.7%) | Wins: 17.0% | eps:  0.327
 180 | 15:11:19 | Agent: -19.1% (-12.5%) | Market:   9.7% ( 13.2%) | Wins: 18.0% | eps:  0.287
 190 | 15:12:40 | Agent: -18.6% (-20.2%) | Market:   9.9% (  9.6%) | Wins: 18.0% | eps:  0.248
 200 | 15:13:59 | Agent: -17.3% ( -8.8%) | Market:   8.9% (  1.3%) | Wins: 22.0% | eps:  0.208
 210 | 15:15:15 | Agent: -16.1% (-10.4%) | Market:   8.8% (  8.6%) | Wins: 25.0% | eps:  0.168
 220 | 15:16:34 | Agent: -16.4% (-21.0%) | Market:   7.2% (  6.9%) | Wins: 25.0% | eps:  0.129
 230 | 15:17:55 | Agent: -15.3% (-10.0%) | Market:   7.4% (  9.6%) | Wins: 25.0% | eps:  0.089
 240 | 15:19:11 | Agent: -14.9% (-10.7%) | Market:   8.2% ( 21.2%) | Wins: 23.0% | eps:  0.050
 250 | 15:20:29 | Agent: -14.5% ( -7.5%) | Market:   7.7% (  2.1%) | Wins: 22.0% | eps:  0.010
 260 | 15:21:46 | Agent: -13.7% (-11.0%) | Market:   9.7% ( 14.9%) | Wins: 19.0% | eps:  0.009
 270 | 15:23:06 | Agent: -11.3% ( -0.5%) | Market:  10.3% ( 15.2%) | Wins: 22.0% | eps:  0.008
 280 | 15:24:27 | Agent: -10.6% ( -5.5%) | Market:  10.3% ( 13.7%) | Wins: 23.0% | eps:  0.007
 290 | 15:25:48 | Agent: -10.0% (-14.9%) | Market:  11.9% ( 25.5%) | Wins: 22.0% | eps:  0.007
 300 | 15:27:13 | Agent:  -9.8% ( -6.2%) | Market:  13.0% ( 12.1%) | Wins: 18.0% | eps:  0.006
 310 | 15:28:35 | Agent: -10.1% (-13.4%) | Market:  14.3% ( 21.4%) | Wins: 15.0% | eps:  0.005
 320 | 15:29:58 | Agent:  -8.7% ( -7.7%) | Market:  14.3% (  6.8%) | Wins: 15.0% | eps:  0.005
 330 | 15:31:22 | Agent:  -7.4% (  3.4%) | Market:  13.9% (  6.4%) | Wins: 16.0% | eps:  0.004
 340 | 15:32:49 | Agent:  -6.4% ( -0.3%) | Market:  11.4% ( -4.5%) | Wins: 19.0% | eps:  0.004
 350 | 15:34:17 | Agent:  -5.8% ( -1.8%) | Market:  11.7% (  5.4%) | Wins: 20.0% | eps:  0.004
 360 | 15:35:48 | Agent:  -5.9% (-11.7%) | Market:  10.5% (  2.9%) | Wins: 23.0% | eps:  0.003
 370 | 15:37:20 | Agent:  -7.1% (-12.5%) | Market:  10.3% ( 13.2%) | Wins: 20.0% | eps:  0.003
 380 | 15:38:57 | Agent:  -5.6% (  8.7%) | Market:   8.3% ( -6.7%) | Wins: 22.0% | eps:  0.003
 390 | 15:40:23 | Agent:  -5.3% (-11.5%) | Market:   6.9% ( 11.8%) | Wins: 23.0% | eps:  0.002
    """
