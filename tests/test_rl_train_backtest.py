"""
Compares data used in training a reinforcement learning model with data used in backtesting
"""
from unittest import TestCase

import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px

from ong_trading.ML.RL.rl_train import create_tradingenv_ddqn, Evaluator
from ong_trading.ML.RL.config import ModelConfig
from ong_trading.ML.RL.rl_backtest import create_train_test_backtesters
from ong_trading.vectorized.pnl import pnl_positions


class Test(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = tf.keras.models.load_model(ModelConfig.model_path(True))
        cls.trading_env, cls.ddqna = create_tradingenv_ddqn()
        # For testing commission will be 0
        cls.bt_train, cls.bt_test = create_train_test_backtesters(commission_rel=0, cash=100)
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

    def iter_backtesting(self, check_train_data=False, check_test_data=True) -> tuple:
        """Yields data_type, backtesting suite and backtesting result (a portfolio object)"""
        bt_suites = dict()
        if check_train_data:
            bt_suites['train'] = self.bt_train
        if check_test_data:
            bt_suites['test'] = self.bt_test
        for k, bt in bt_suites.items():
            # Backtesting must be run to make sure all data is initialized
            portfolio = bt.run(print_debug_msg=False)
            yield k, bt, portfolio

    def test_same_data(self, check_train_data=False, check_test_data=True):
        """
        Tests that data used for backtesting and training is the same. It checks original data, features
        and model output data
        :param check_train_data: if True, data for train period will be used (defaults False)
        :param check_test_data: if True, data for test period will be used (defaults to True
        :return: None
        """
        for data_type, bt, backtesting_ptf in self.iter_backtesting(check_train_data=check_train_data,
                                                                    check_test_data=check_test_data):
            training_features = self.trading_env.data_source.features(data_type)
            training_data = self.trading_env.data_source.data(data_type)
            backtesting_data = pd.DataFrame(bt.strategy.model_data
                                            ).set_index("timestamp").loc[:, training_data.columns]
            backtesting_features = pd.DataFrame(np.vstack(bt.strategy.model_features), index=bt.strategy.model_dates)
            #############################################################################
            # Comparison of original data
            #############################################################################
            # Note that when features need some initial data, it might happen than the length of both
            # vectors are the same, so we are comparing just the values of the potentially shorter one (backtesting)
            self.assertTrue(backtesting_data.equals(training_data.iloc[-len(backtesting_data):]),
                            "Original data is not the same")
            #############################################################################
            # Comparison of features (model input data)
            #############################################################################
            self.assertTrue((backtesting_features.index == training_features.index).all(),
                            "Indexes of processed input data do not match")
            #############################################################################
            # Comparison of model input data
            #############################################################################
            self.assertTrue(np.allclose(backtesting_features.values,
                                        training_features.values,
                                        atol=1e-6),
                            "Model inputs do not match")
            #############################################################################
            # Comparison of model out data
            #############################################################################
            backtesting_model_outputs = np.vstack(bt.strategy.model_outputs)
            training_model_outputs = self.ddqna.online_network(training_features.values)
            self.assertTrue(np.allclose(backtesting_model_outputs, training_model_outputs),
                            "Model outputs do not match")

    def test_same_results(self, check_train_data=False, check_test_data=True):
        """
        Tests that results calculated in backtesting and training is the same.
        :param check_train_data: if True, data for train period will be used (defaults False)
        :param check_test_data: if True, data for test period will be used (defaults to True
        :return: None
        """
        training_res = Evaluator(self.ddqna, self.trading_env.data_source)
        for data_type, bt, backtesting_ptf in self.iter_backtesting(check_train_data=check_train_data,
                                                                    check_test_data=check_test_data):
            training_output = training_res.output_analyser[data_type]['strategy']
            backtesting_ptf.create_equity_curve_dataframe()
            backtesting_output = backtesting_ptf.get_analysis()
            ##############################
            #   Check positions
            ##############################
            backtesting_pos = np.sign(pd.DataFrame(backtesting_ptf.all_positions).iloc[:, 0].values)
            training_pos = training_res.positions[data_type]
            self.assertTrue(np.allclose(backtesting_pos[1:], training_pos),
                            "Positions are not the same")
            ###################################
            #   Check pnl (using returns)
            ###################################
            self.assertTrue(np.allclose(backtesting_ptf.equity_curve['returns'].values[1:],
                                        backtesting_output.returns[1:]),
                            "Backtesting portfolio and output analyser give different results")
            # Just to check both charts are the same
            backtesting_ptf.output_summary_stats(plot=True)
            bt_ret = backtesting_output.returns[1:]
            tr_ret = training_output.returns
            zero_rets = abs(tr_ret) < 1e-9
            self.assertTrue(np.allclose(bt_ret[zero_rets], tr_ret[zero_rets]),
                            "Zero returns are not the same")
            # Non zero returns must be proportional
            non_zero_rets = ~ zero_rets
            factors = bt_ret[non_zero_rets] / tr_ret[non_zero_rets]
            fig = px.line(factors, title="Factors")
            fig.show()
            self.assertTrue((factors.max() - factors.min()) / factors.mean() < .001,
                            "Calculated returns are not proportional")
            pass

    def test_pnl_backtesting(self):
        """Checks that pnl of backtesting is correctly calculated"""
        for data_type, bt, backtesting_ptf in self.iter_backtesting(check_train_data=True,
                                                                    check_test_data=True):
            backtesting_pos = pd.DataFrame(backtesting_ptf.all_positions).iloc[:, 0].values
            backtesting_output = backtesting_ptf.get_analysis()
            backtesting_prices = bt.data.to_pandas("ELE.MC")[bt.start_date:bt.end_date]
            calculated_pnl = pnl_positions(backtesting_pos[-len(backtesting_prices):],
                                           backtesting_prices.close, backtesting_prices.close)
            backtesting_pnl = pd.DataFrame(backtesting_ptf.all_holdings).set_index("datetime")['ELE.MC']
            self.assertTrue(np.allclose(backtesting_pnl.iloc[-len(calculated_pnl):], calculated_pnl),
                            f"Pnl with two different calculations do not match for {data_type}")
            backtesting_output_pnl = backtesting_output.cash_curve - backtesting_output.cash_curve[0]
            self.assertTrue(np.allclose(backtesting_pnl, backtesting_output_pnl),
                            f"Pnl of portfolio and output do not match for {data_type}")
