"""
Tests that compare the data used in training a network with data used in backtesting
"""
from unittest import TestCase

import tensorflow as tf
import numpy as np
import pandas as pd

from ong_trading.ML.RL.rl_train import create_tradingenv_ddqn
from ong_trading.ML.RL.config import ModelConfig
from ong_trading.ML.RL.rl_test import create_train_test_backtesters


class Test(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = tf.keras.models.load_model(ModelConfig.model_path(True))
        cls.trading_env, cls.ddqna = create_tradingenv_ddqn()
        cls.bt_train, cls.bt_test = create_train_test_backtesters()
        cls.bt_test.reset()

    def test_same_models(self):
        """Tests that networks (the ones stored) used for training the model and backtesting are the same"""
        self.assertTrue(self.model.get_config() == self.ddqna.online_network.get_config(),
                        "Model used for comparing backtesting and training are not the same")

        # These two networks cannot be the same because one is trainable and the other is not, even it their weights
        # can be the same in certain moments during model training
        self.assertFalse(self.ddqna.online_network.get_config() == self.ddqna.target_network.get_config(),
                         "Model used for online and target the same (one is trainable and the other is not)")

        self.assertTrue(self.bt_test.strategy.model.get_config() == self.ddqna.online_network.get_config(),
                        "Model used for training and backtesting are not the same")

    def test_same_preprocessors(self):
        """Tests that scalers parameters (the ones stored) used for training the model and backtesting are the same"""
        preprocessor_train = self.trading_env.data_source.preprocessor
        preprocessor_bt = self.bt_test.strategy.preprocessor

        self.assertTrue(preprocessor_train == preprocessor_bt,
                        "Preprocessor used for comparing backtesting and training are not the same")

    def test_train_same_data(self, check_train_data=False, check_test_data=True):
        """
        Tests that data used for backtesting and training is the same. It checks original data, input data
        and model output data
        :param check_train_data: if True, data for train period will be used (defaults False)
        :param check_test_data: if True, data for test period will be used (defaults to True
        :return: None
        """
        bt_suites = dict()
        if check_train_data:
            bt_suites['train'] = self.bt_train
        if check_test_data:
            bt_suites['test'] = self.bt_test
        for data_type, bt in bt_suites.items():
            # run backtest on test data for force initialize data
            res = bt.run(print_debug_msg=False)
            training_features = self.trading_env.data_source.features(data_type)
            training_data = self.trading_env.data_source.data(data_type)

            backtesting_data = pd.DataFrame(bt.strategy.model_data
                                            ).set_index("timestamp").loc[:, training_data.columns]
            backtesting_features = pd.DataFrame(np.vstack(bt.strategy.model_features), index=bt.strategy.model_dates)
            self.assertTrue((backtesting_features.index == training_features.index).all(),
                            "Indexes of processed input data do not match")
            self.assertTrue(backtesting_data.equals(training_data.iloc[-len(backtesting_data):]),
                            "Original data is not the same")
            self.assertTrue(np.allclose(backtesting_features.values,
                                        training_features.values,
                                        atol=1e-6),
                            "Model inputs do not match")
            # Now, check model outputs
            backtesting_model_outputs = np.vstack(bt.strategy.model_outputs)
            training_model_outputs = self.ddqna.online_network(training_features.values)
            self.assertTrue(np.allclose(backtesting_model_outputs, training_model_outputs),
                            "Model outputs do not match")
