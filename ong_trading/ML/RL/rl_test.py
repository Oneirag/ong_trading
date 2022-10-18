from pathlib import Path
from time import time

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

import tensorflow as tf

import gym
from gym.envs.registration import register

from ong_trading.ML.RL.ddqna_agent import DDQNAgent

try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    # Invalid device or cannot modify virtual devices once initialized.
    pass

np.random.seed(42)
tf.random.set_seed(42)


# Check if using GPU
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    print('Using GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)
else:
    print('Using CPU')

results_path = Path('results', 'trading_bot')
if not results_path.exists():
    results_path.mkdir(parents=True)


# Helper function
def format_time(t):
    m_, s = divmod(t, 60)
    h, m = divmod(m_, 60)
    return '{:02.0f}:{:02.0f}:{:02.0f}'.format(h, m, s)


# Set up environment
trading_days = 252
register(
    id='trading-v0',
    entry_point='ong_trading.ML.RL.trading_env:TradingEnvironment',
    max_episode_steps=trading_days,
    kwargs=dict(ticker="ELE.MC")
)

# Initialize trading environment
trading_environment = gym.make('trading-v0') #, ticker='ELE.MC')
trading_environment.env.trading_days = trading_days
trading_environment.env.trading_cost_bps = 1e-3
trading_environment.env.time_cost_bps = 1e-4
trading_environment.seed(42)

# Get environ params
state_dim = trading_environment.observation_space.shape[0]
num_actions = trading_environment.action_space.n
max_episode_steps = trading_environment.spec.max_episode_steps



#Define hyperparameters
gamma = .99,  # discount factor
tau = 100  # target network update frequency
#NN Architecture
architecture = (256, 256)  # units per layer
learning_rate = 0.0001  # learning rate
l2_reg = 1e-6  # L2 regularization
#Experience Replay
replay_capacity = int(1e6)
batch_size = 4096
#ϵ-greedy Policy
epsilon_start = 1.0
epsilon_end = .01
epsilon_decay_steps = 250
epsilon_exponential_decay = .99
#Create DDQN Agent
#We will use TensorFlow to create our Double Deep Q-Network .

tf.keras.backend.clear_session()
ddqn = DDQNAgent(state_dim=state_dim,
                 num_actions=num_actions,
                 learning_rate=learning_rate,
                 gamma=gamma,
                 epsilon_start=epsilon_start,
                 epsilon_end=epsilon_end,
                 epsilon_decay_steps=epsilon_decay_steps,
                 epsilon_exponential_decay=epsilon_exponential_decay,
                 replay_capacity=replay_capacity,
                 architecture=architecture,
                 l2_reg=l2_reg,
                 tau=tau,
                 batch_size=batch_size)
ddqn.online_network.summary()

#Set parameters
total_steps = 0
max_episodes = 1000
# max_episodes = 100       # Just for testing
max_episodes = 50       # Just for testing
# Initialize variables
episode_time, navs, market_navs, diffs, episode_eps = [], [], [], [], []


# Visualization
def track_results(episode, nav_ma_100, nav_ma_10,
                  market_nav_100, market_nav_10,
                  win_ratio, total, epsilon):
    # time_ma = np.mean([episode_time[-100:]])
    # T = np.sum(episode_time)

    template = '{:>4d} | {} | Agent: {:>6.1%} ({:>6.1%}) | '
    template += 'Market: {:>6.1%} ({:>6.1%}) | '
    template += 'Wins: {:>5.1%} | eps: {:>6.3f}'
    print(template.format(episode, format_time(total),
                          nav_ma_100 - 1, nav_ma_10 - 1,
                          market_nav_100 - 1, market_nav_10 - 1,
                          win_ratio, epsilon))


# Train Agent
if __name__ == '__main__':
    np.seterr(all='raise')

    start = time()
    results = []
    for episode in range(1, max_episodes + 1):
        this_state = trading_environment.reset()
        for episode_step in range(max_episode_steps):
            action = ddqn.epsilon_greedy_policy(this_state.reshape(-1, state_dim))
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
            track_results(episode, np.mean(navs[-100:]), np.mean(navs[-10:]),
                          np.mean(market_navs[-100:]), np.mean(market_navs[-10:]),
                          np.sum([s > 0 for s in diffs[-100:]]) / min(len(diffs), 100),
                          time() - start, ddqn.epsilon)
        if len(diffs) > 25 and all([r > 0 for r in diffs[-25:]]):
            print(result.tail())
            break

    trading_environment.close()


    # Store Results
    results = pd.DataFrame({'Episode': list(range(1, episode+1)),
                            'Agent': navs,
                            'Market': market_navs,
                            'Difference': diffs}).set_index('Episode')

    results['Strategy Wins (%)'] = (results.Difference > 0).rolling(100).sum()
    results.info()

    with sns.axes_style('white'):
        # sns.distplot(results.Difference)
        sns.histplot(results.Difference)
        sns.despine()

    fig, axes = plt.subplots(ncols=2, figsize=(14, 4), sharey=True)

    df1 = (results[['Agent', 'Market']]
           .sub(1)
           .rolling(100)
           .mean())
    df1.plot(ax=axes[0],
             title='Annual Returns (Moving Average)',
             lw=1)

    df2 = results['Strategy Wins (%)'].div(100).rolling(50).mean()
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
