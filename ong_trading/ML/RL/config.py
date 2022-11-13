"""
General configuration of the Reinforcement learning model:
- ModelHyperParams: hyperparameters and network architecture
- ModelConfig: model name, # of episodes, ticker, path for saving, max date for train...
"""
from dataclasses import dataclass, field
from ong_trading.ML import get_model_path as gmp
from ong_trading.features.preprocess import PCA_Preprocessor, RLPreprocessorClose, MLPreprocessor
from functools import partial


@dataclass
class ModelHyperParamsBase:
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


class ModelHyperParamsLowerGamma(ModelHyperParamsBase):
    gamma = 0.95


@dataclass
class ModelConfigBase:
    trading_cost_bps = 1e-3
    time_cost_bps = 1e-4
    random_seed = 42
    trading_days = 252
    ticker = "ELE.MC"
    # ticker = "SAN.MC"
    _model_name = ""     # Is added to the ticker to create the model name
    # Number of training episodes
    max_episodes = 1000
    max_episodes = 1500

    train_start = "2020-01-01"
    train_start = None
    validation_start = None
    test_start = "2022-01-01"

    preprocessor: MLPreprocessor = None

    @classmethod
    @property
    def full_model_name(cls) -> str:
        return cls.ticker + "_" + cls._model_name

    @classmethod
    def model_path(cls, extra):
        return gmp(cls.full_model_name, extra)


class ModelConfigIndicators(ModelConfigBase):
    _model_name = "indicadores"
    preprocessor = partial(RLPreprocessorClose, normalize=True)


class ModelConfigPCA(ModelConfigBase):
    _model_name = "pca"
    preprocessor = partial(PCA_Preprocessor, symbol_name=ModelConfigBase.ticker)


ModelConfig = ModelConfigIndicators
# ModelConfig = ModelConfigPCA
ModelHyperParams = ModelHyperParamsBase
ModelHyperParams = ModelHyperParamsLowerGamma
