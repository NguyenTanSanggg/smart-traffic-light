import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from agents.replay_buffer import ReplayBuffer

class QNetwork(Model):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = Dense(256, activation='relu')
        self.fc2 = Dense(256, activation='relu')
        self.out = Dense(action_size)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)

class DQN:
    def __init__(self, state_size, action_size,
                 gamma=0.99, lr=1e-4, buffer_size=100000):
        self.q = QNetwork(state_size, action_size)
        self.target_q = QNetwork(state_size, action_size)
        self.target_q.set_weights(self.q.get_weights())

        self.opt = Adam(lr)
        self.replay = ReplayBuffer(buffer_size)

        self.steps_done = 0
        self.action_size = action_size
        self.gamma = gamma
        self.eps_start, self.eps_end, self.eps_decay = 1.0, 0.05, 200000

    def select_action(self, state):
        eps = self.eps_end + (self.eps_start - self.eps_end) * \
              np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if np.random.rand() < eps:
            return np.random.randint(self.action_size)
        state = np.expand_dims(state, axis=0).astype(np.float32)
        q_values = self.q(state)
        return int(tf.argmax(q_values[0]).numpy())

    def select_action_greedy(self, state):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        q_values = self.q(state)
        return int(tf.argmax(q_values[0]).numpy())

    def optimize(self, batch_size=64):
        if len(self.replay) < batch_size:
            return

        batch = self.replay.sample(batch_size)
        state = np.vstack(batch.state).astype(np.float32)
        action = np.array(batch.action, dtype=np.int32)
        reward = np.array(batch.reward, dtype=np.float32)
        next_state = np.vstack(batch.next_state).astype(np.float32)
        done = np.array(batch.done, dtype=np.float32)

        with tf.GradientTape() as tape:
            q_values = tf.reduce_sum(
                self.q(state) * tf.one_hot(action, self.action_size), axis=1
            )
            q_next = tf.reduce_max(self.target_q(next_state), axis=1)
            q_target = reward + self.gamma * q_next * (1 - done)
            loss = tf.reduce_mean(tf.square(q_target - q_values))

        grads = tape.gradient(loss, self.q.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.q.trainable_variables))

    def update_target(self):
        self.target_q.set_weights(self.q.get_weights())
