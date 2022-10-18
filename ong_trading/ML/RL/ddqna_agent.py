import os
# import pickle
from collections import deque
from random import sample

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


class DDQNAgent:
    """
    Definition of the trading agent
    """

    def __init__(self, state_dim,
                 num_actions,
                 learning_rate,
                 gamma,
                 epsilon_start,
                 epsilon_end,
                 epsilon_decay_steps,
                 epsilon_exponential_decay,
                 replay_capacity,
                 architecture,
                 l2_reg,
                 tau,
                 batch_size):

        self.state_dim = state_dim
        self.num_actions = num_actions
        self.experience = deque([], maxlen=replay_capacity)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.architecture = architecture
        self.l2_reg = l2_reg

        self.model_weights_path = os.path.join("results", "model_weights.tf")

        self.online_network = self.build_model()
        self.target_network = self.build_model(trainable=False)
        self.update_target(store_weights=False)

        self.epsilon = epsilon_start
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history = []

        self.total_steps = self.train_steps = 0
        self.episodes = self.episode_length = self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []

        self.batch_size = batch_size
        self.tau = tau
        self.losses = []
        self.idx = tf.range(batch_size, dtype=np.int64)
        self.train = True

    def build_model(self, trainable=True):
        layers = []
        n = len(self.architecture)
        for i, units in enumerate(self.architecture, 1):
            layers.append(Dense(units=units,
                                input_dim=self.state_dim if i == 1 else None,
                                activation='relu',
                                kernel_regularizer=l2(self.l2_reg),
                                name=f'Dense_{i}',
                                trainable=trainable))
        layers.append(Dropout(.1))
        layers.append(Dense(units=self.num_actions,
                            trainable=trainable,
                            name='Output'))
        model = Sequential(layers)
        model.compile(loss='mean_squared_error',
                      optimizer=Adam(learning_rate=self.learning_rate))
        # Load from disk if there is any
        if os.path.isfile(self.model_weights_path):
            model.load_weights(self.model_weights_path)
            # with open(self.model_weights_path, "rb") as f:
            #     weights = pickle.load(f)
            # model.set_weights(weights)
        return model

    def update_target(self, store_weights=True):
        """Updates target network but also stores weights if store_weights=True"""
        weights = self.online_network.get_weights()
        self.target_network.set_weights(weights)
        if store_weights:
            self.target_network.save_weights(self.model_weights_path, save_format="tf", overwrite=True)
            # with open(self.model_weights_path, "wb") as f:
            #     pickle.dump(weights, f, protocol=4)

    def epsilon_greedy_policy(self, state):
        self.total_steps += 1
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        q = self.online_network.predict(state, verbose=0)
        # q = self.online_network.predict_on_batch(state.reshape(1, self.state_dim))
        # q = self.online_network(state.reshape(1, self.state_dim), training=False)
        return np.argmax(q, axis=1).squeeze()

    def memorize_transition(self, s, a, r, s_prime, not_done):
        if not_done:
            self.episode_reward += r
            self.episode_length += 1
        else:
            if self.train:
                if self.episodes < self.epsilon_decay_steps:
                    self.epsilon -= self.epsilon_decay
                else:
                    self.epsilon *= self.epsilon_exponential_decay

            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward, self.episode_length = 0, 0

        self.experience.append((s, a, r, s_prime, not_done))

    def experience_replay(self):
        if self.batch_size > len(self.experience):
            return
        minibatch = map(np.array, zip(*sample(self.experience, self.batch_size)))
        states, actions, rewards, next_states, not_done = minibatch

        next_q_values = self.online_network.predict_on_batch(next_states)
        best_actions = tf.argmax(next_q_values, axis=1)

        next_q_values_target = self.target_network.predict_on_batch(next_states)
        target_q_values = tf.gather_nd(next_q_values_target,
                                       tf.stack((self.idx, best_actions), axis=1))

        targets = rewards + not_done * self.gamma * target_q_values

        q_values = self.online_network.predict_on_batch(states)
        q_values[(self.idx, actions)] = targets

        loss = self.online_network.train_on_batch(x=states, y=q_values)
        self.losses.append(loss)

        if self.total_steps % self.tau == 0:
            self.update_target()


if __name__ == '__main__':
    from rl_test import (state_dim, num_actions, learning_rate, gamma, epsilon_start, epsilon_end,
                         epsilon_decay_steps, epsilon_exponential_decay, replay_capacity, architecture, l2_reg,
                         tau, batch_size)

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
