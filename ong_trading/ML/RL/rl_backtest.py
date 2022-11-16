"""
Executes backtester on rl model
"""

from functools import partial
import pandas as pd

from ong_trading.event_driven.backtester import Backtester
from ong_trading.strategy.strategy import MachineLearningStrategy
from ong_trading.features.preprocess import MLPreprocessor
from ong_trading.ML.RL.config import ModelConfig
from ong_trading.event_driven.execution import SimulatedBroker
from ong_trading.event_driven.portfolio import ConstantSizeNaivePortfolio


def create_train_test_backtesters(commission_rel=None):
    test_kwargs = dict(
        cash=7000, symbol_list=ModelConfig.ticker, strategy_class=MachineLearningStrategy,
        start_date=pd.Timestamp(ModelConfig.test_start),
        strategy_kwargs=dict(model_path=ModelConfig.model_path(True),
                             preprocessor=MLPreprocessor.load(
                                 ModelConfig.model_path("preprocessor"))
                             ),
        portfolio_class=ConstantSizeNaivePortfolio,
    )
    if commission_rel is not None:
        test_kwargs['broker_class'] = partial(SimulatedBroker, commission_rel=commission_rel)
    train_kwargs = test_kwargs.copy()
    train_kwargs['end_date'] = ModelConfig.validation_start or ModelConfig.test_start
    train_kwargs['start_date'] = ModelConfig.train_start

    bt_test = bt_train = None
    bt_test = Backtester(**test_kwargs)
    bt_train = Backtester(**train_kwargs)

    return bt_train, bt_test


if __name__ == '__main__':

    bt_train, bt_test = create_train_test_backtesters()

    for bt, extra_name in (bt_test, "test"), (bt_train, "train"):
        if bt is not None:
            result = bt.run(print_debug_msg=False)
            print(result.output_summary_stats(plot=True, model_name=ModelConfig.full_model_name + " " + extra_name))
