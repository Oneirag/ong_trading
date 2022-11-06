import os
# import pickle
# from collections import deque
# from random import sample
import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from ong_utils import OngTimer
from ong_trading.ML import get_model_path
from ong_trading import logger


class Experience:
    """Class to speed up Experience replay by storing into numpy arrays of max_size rows instead of a dequeue"""
    def __init__(self, max_size):
        self.max_len = max_size
        self.length = 0
        self.idx = 0
        self.np_s = None
        self.np_a = None
        self.np_r = None
        self.np_s_prime = None
        self.np_not_done = None

    def __init_np_like(self, x, dtype=np.float64):
        """Returns an array of the same columns as x and self.max_len rows"""
        if isinstance(x, (int, float, np.int64)):
            return np.empty(self.max_len, dtype=dtype)
        else:
            return np.empty((self.max_len, len(x)), dtype=dtype)

    def append(self, s, a, r, s_prime, not_done):
        if self.np_s is None:
            self.np_s = self.__init_np_like(s)
            self.np_a = self.__init_np_like(a, int)
            self.np_r = self.__init_np_like(r)
            self.np_s_prime = self.__init_np_like(s_prime)
            self.np_not_done = self.__init_np_like(not_done, int)
        self.np_s[self.idx] = s
        self.np_a[self.idx] = a
        self.np_r[self.idx] = r
        self.np_s_prime[self.idx] = s_prime
        self.np_not_done[self.idx] = not_done
        self.length = min(self.length + 1, self.max_len)
        self.idx = (self.idx + 1) % self.max_len

    def __len__(self):
        return self.length

    def minibatch(self, batch_size) -> tuple:
        """Returns a random minibatch of states, actions, rewards, next_states, not_done"""
        choices = np.random. default_rng().choice(range(self.length), batch_size, replace=False)
        states = self.np_s[choices]
        actions = self.np_a[choices]
        rewards = self.np_r[choices]
        next_states = self.np_s_prime[choices]
        not_done = self.np_not_done[choices]
        return states, actions, rewards, next_states, not_done


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
                 batch_size,
                 model_name="model1",
                 random_seed=None):

        if random_seed:
            np.random.seed(random_seed)

        self.timer = OngTimer(logger=logger)

        self.state_dim = state_dim
        self.num_actions = num_actions
        # self.old_experience = deque([], maxlen=replay_capacity)
        self.experience = Experience(max_size=replay_capacity)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.architecture = architecture
        self.l2_reg = l2_reg

        self.model_name = model_name

        self.online_network = self.build_model()
        self.target_network = self.build_model(trainable=False)
        self.update_target()

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

    def print_timers(self):
        """Prints all opened timers (for time debugging)"""
        for timer_msg in self.timer.msgs:
            try:
                self.timer.print_loop(timer_msg)
            except:
                pass

    def build_model(self, trainable=True):
        # Load model from disk (if there is any)
        model_path = get_model_path(self.model_name, trainable)
        if os.path.isdir(model_path):
            with self.timer.context_manager(f"Loading saved model from {model_path}"):
                model = tf.keras.models.load_model(model_path)
        else:
            # No model found. Build it from scratch
            layers = []
            n = len(self.architecture)
            for i, units in enumerate(self.architecture, 1):
                layers.append(Dense(units=units,
                                    input_dim=self.state_dim if i == 1 else None,
                                    activation='relu',
                                    kernel_regularizer=l2(self.l2_reg),
                                    name=f'dense_{i}',
                                    trainable=trainable))
            layers.append(Dropout(.1))
            layers.append(Dense(units=self.num_actions,
                                trainable=trainable,
                                name='output'))
            model = Sequential(layers)
            model.compile(loss='mean_squared_error',
                          optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target(self):
        """Updates target network"""
        weights = self.online_network.get_weights()
        self.target_network.set_weights(weights)

    def epsilon_greedy_policy(self, state):
        self.total_steps += 1
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.num_actions)
        # q = self.online_network.predict(state, verbose=0)
        q = self.online_network.predict_on_batch(state.reshape(1, self.state_dim))
        # q = self.online_network(state.reshape(1, self.state_dim), training=False)
        return np.argmax(q, axis=1).squeeze()

    def epsilon_greedy_policy_all(self, states, training=True):
        """Get actions for all steps at once"""
        n_states = states.shape[0]
        retval = np.empty(n_states, dtype=np.integer)
        self.total_steps += n_states
        # If training=False epsilon is truncated to 0 so network can be evaluated without random choices
        epsilon = self.epsilon if training else 0
        # epsilon_mask is true where random choice has to be used.
        epsilon_mask = np.random.random(n_states) <= epsilon
        retval[epsilon_mask] = np.random.choice(self.num_actions, sum(epsilon_mask))
        if n_states * (1 - epsilon) < 4000:       # below this level is faster to use __call__ than predict on batch
            qs = self.online_network(states[~epsilon_mask, :], training=False)
        else:
            qs = self.online_network.predict_on_batch(states[~epsilon_mask, :])
        retval[~epsilon_mask] = np.argmax(qs, axis=1).squeeze()
        return retval

    def update_epsilon(self):
        if self.episodes < self.epsilon_decay_steps:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon *= self.epsilon_exponential_decay

    def memorize_transition(self, s, a, r, s_prime, not_done):
        if not_done:
            self.episode_reward += r
            self.episode_length += 1
        else:
            if self.train:
                self.update_epsilon()
            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward, self.episode_length = 0, 0

        self.experience.append(*(s, a, r, s_prime, not_done))
        # self.old_experience.append((s, a, r, s_prime, not_done))

    def experience_replay(self):
        if self.batch_size > len(self.experience):
            return
        self.timer.tic("sample")
        # minibatch = map(np.array, zip(*sample(self.old_experience, self.batch_size)))
        # states, actions, rewards, next_states, not_done = minibatch
        states, actions, rewards, next_states, not_done = self.experience.minibatch(self.batch_size)
        self.timer.toc_loop("sample")

        self.timer.tic("online_network_prediction")
        # Replace this two individual calls to online network with a single one for both (for faster speed)
        # next_q_values = self.online_network.predict_on_batch(next_states)
        # q_values = self.online_network.predict_on_batch(states)
        online_values = self.online_network.predict_on_batch(np.vstack([next_states, states]))
        next_q_values, q_values = np.split(online_values, 2)
        best_actions = tf.argmax(next_q_values, axis=1)
        self.timer.toc_loop("online_network_prediction")

        self.timer.tic("target_network_prediction")
        next_q_values_target = self.target_network.predict_on_batch(next_states)
        target_q_values = tf.gather_nd(next_q_values_target,
                                       tf.stack((self.idx, best_actions), axis=1))
        self.timer.toc_loop("target_network_prediction")

        targets = rewards + not_done * self.gamma * target_q_values

        # q_values = self.online_network(states, training=False).numpy()
        q_values[(self.idx, actions)] = targets

        self.timer.tic("online_network_training")
        loss = self.online_network.train_on_batch(x=states, y=q_values)

        self.losses.append(loss)
        self.timer.toc_loop("online_network_training")

        if self.total_steps % self.tau == 0:
            self.update_target()

    def save_model(self):
        """Saves target_network for evaluation just calling tf.keras.models.load_model"""
        model_path = get_model_path(self.model_name, False)
        train_model_path = get_model_path(self.model_name, True)
        tic_msg = f"Saving model to {model_path} and {train_model_path}"
        self.timer.tic(tic_msg)
        self.target_network.save(model_path)
        self.online_network.save(train_model_path)
        self.timer.toc_loop(tic_msg)


if __name__ == '__main__':

    exp = Experience(max_size=10)
    s = np.arange(10)
    a = 1
    r = 1
    s_prime = np.arange(10)
    not_done = 1
    for _ in range(15):
        exp.append(s, a, r, s_prime, not_done)
        s += 1
        a += 1
        r += 1
        s_prime -= 1
        not_done -= 1

    print(exp.np_a)
    print(exp.np_s)
    print(exp.minibatch(10))
    print(exp.minibatch(10))
    print(exp.minibatch(10))

    from rl_train import (state_dim, num_actions, learning_rate, gamma, epsilon_start, epsilon_end,
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
