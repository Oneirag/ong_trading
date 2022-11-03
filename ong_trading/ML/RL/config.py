"""
General configuration of the Reinforcement learning model:
- ModelHyperParams: hyperparameters and network architecture
- ModelConfig: model name, # of episodes, ticker, path for saving, max date for train...
"""
from dataclasses import dataclass, field
from ong_trading.ML import get_model_path as gmp

@dataclass
class ModelHyperParams:
    # Define hyperparameters for ddqn network
    gamma: field() = .99,  # discount factor
    tau: field() = 100  # target network update frequency
    # NN Architecture
    architecture: field() = (256, 256)  # units per layer
    learning_rate: field() = 0.0001  # learning rate
    l2_reg: field() = 1e-6  # L2 regularization
    # Experience Replay
    replay_capacity: field() = int(1e6)
    batch_size: field() = 4096
    # ϵ-greedy Policy
    epsilon_start: field() = 1.0
    epsilon_end: field() = .01
    epsilon_decay_steps: field() = 250
    epsilon_exponential_decay: field() = .99


@dataclass
class ModelConfig:
    random_seed = 42
    trading_days = 252
    ticker = "ELE.MC"
    model_name = "ELE.MC_indicadores"
    # Number of training episodes
    max_episodes = 1000
    max_episodes = 1500
    # A date for which validation will start (so training will NOT use this date)
    train_split_data = "2022-01-01"

    @classmethod
    def model_path(cls, extra):
        return gmp(cls.model_name, extra)

