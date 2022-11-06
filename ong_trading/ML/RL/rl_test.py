"""
Executes backtester on rl model
"""

import pandas as pd


from ong_trading.event_driven.backtester import Backtester
from ong_trading.strategy.strategy import MachineLearningStrategy
from ong_trading.features.preprocess import MLPreprocessor
from ong_trading.ML.RL.config import ModelConfig


if __name__ == '__main__':
    test_kwargs = dict(
        cash=7000, symbol_list=ModelConfig.ticker, strategy_class=MachineLearningStrategy,
        start_date=pd.Timestamp(ModelConfig.train_split_date),
        strategy_kwargs=dict(model_path=ModelConfig.model_path(True),
                             preprocessor=MLPreprocessor.load(
                                 ModelConfig.model_path("preprocessor"))
                             )
    )
    train_kwargs = test_kwargs.copy()
    train_kwargs['end_date'] = ModelConfig.train_split_date
    train_kwargs['start_date'] = ModelConfig.train_start_date

    bt_test = bt_train = None
    bt_test = Backtester(**test_kwargs)
    bt_train = Backtester(**train_kwargs)

    for bt, extra_name in (bt_test, "test"), (bt_train, "train"):
        if bt is not None:
            result = bt.run(print_debug_msg=False)
            print(result.output_summary_stats(plot=True, model_name=ModelConfig.model_name + " " + extra_name))

