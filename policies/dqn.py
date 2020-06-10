# dqn policy
import os
import numpy as np
import tensorflow as tf
import datetime

from gym.spaces import Box, Discrete

from policies.memory import Transition, ReplayMemory
from policies.base import Policy, BaseModelMixin, TrainConfig
from utils.misc import plot_learning_curve

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class DefaultConfig(TrainConfig):
    n_episodes = 600
    warmup_episodes = 400

    # fixed learning rate
    learning_rate = 0.001
    end_learning_rate = 0.0001
    # decaying learning rate
    learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=learning_rate,
                                                                  decay_steps=n_episodes,
                                                                  end_learning_rate=learning_rate, power=1.0)
    gamma = 1.0
    epsilon = 1.0
    epsilon_final = 0.01
    memory_size = 100000
    target_update_every_step = 25
    log_every_episode = 10

    batch_normalization = False
    batch_size = 256


class DQNAgent(Policy, BaseModelMixin):

    def __init__(self, env, name, config=None, training=True, layers=(128, 128, 128)):

        Policy.__init__(self, env, name, training=training)
        BaseModelMixin.__init__(self, name)

        self.env = env
        self.config = config
        self.batch_size = self.config.batch_size
        self.memory = ReplayMemory(capacity=self.config.memory_size)
        self.layers = layers

        # Optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.config.learning_rate)#, clipnorm=5)
        # Target net
        self.target_net = tf.keras.Sequential()
        self.target_net.add(tf.keras.layers.Input(shape=env.observation_space.shape))
        for i, layer_size in enumerate(self.layers):
            self.target_net.add(tf.keras.layers.Dense(layer_size, activation='relu'))
            if self.config.batch_normalization:
                self.target_net.add(tf.keras.layers.BatchNormalization())
        self.target_net.add(tf.keras.layers.Dense(env.action_space.n, activation='linear'))
        self.target_net.build()

        self.target_net.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

        # Main net
        self.main_net = tf.keras.Sequential()
        self.main_net.add(tf.keras.layers.Input(shape=env.observation_space.shape))
        for i, layer_size in enumerate(self.layers):
            self.main_net.add(tf.keras.layers.Dense(layer_size, activation='relu'))
            if self.config.batch_normalization:
                self.main_net.add(tf.keras.layers.BatchNormalization())
        self.main_net.add(tf.keras.layers.Dense(env.action_space.n, activation='linear'))
        self.main_net.build()

        self.main_net.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError())

    # def append_sample(self, state, action, reward, next_state, done):
    #     self.memory.append((state, action, reward, next_state, done))

    def get_action(self, state, epsilon):
        q_value = self.main_net.predict_on_batch(state[None, :])
        if np.random.rand() <= epsilon:
            # print('Taking random')
            action = np.random.choice(self.act_size)
        else:
            action = np.argmax(q_value)
        return action, q_value

    def update_target(self):
        self.target_net.set_weights(self.main_net.get_weights())

    def train(self):
        batch = self.memory.sample(self.batch_size)
        states = batch['s']
        actions = batch['a']
        rewards = batch['r']
        next_states = batch['s_next']
        dones = batch['done']

        dqn_variable = self.main_net.trainable_variables
        with tf.GradientTape() as tape:
            tape.watch(dqn_variable)
            # simple dqn
            # target_q = self.target_net.predict_on_batch(next_states)
            # next_action = np.argmax(target_q.numpy(), axis=1)
            # double_dqn
            target_q = self.main_net.predict_on_batch(next_states)
            next_action = np.argmax(target_q.numpy(), axis=1)
            target_q = self.target_net.predict_on_batch(next_states)


            # the next saction should be selected using the online net

            target_value = tf.reduce_sum(tf.one_hot(next_action, self.act_size) * target_q, axis=1)
            target_value = (1 - dones) * self.config.gamma * target_value + rewards

            main_q = self.main_net.predict_on_batch(states)
            main_value = tf.reduce_sum(tf.one_hot(actions, self.act_size) * main_q, axis=1)

            td_error = target_value - main_value
            error = tf.square(td_error) * 0.5
            error = tf.reduce_mean(error)

        # loss = self.main_net.train_on_batch(states, target_value)
        dqn_grads = tape.gradient(error, dqn_variable)
        self.optimizer.apply_gradients(zip(dqn_grads, dqn_variable))
        return error

    def run(self):

        n_episodes = self.config.n_episodes
        warmup_episodes = self.config.warmup_episodes
        epsilon = self.config.epsilon
        epsilon_final = self.config.epsilon_final
        loss = None
        eps_drop = (epsilon - epsilon_final) / warmup_episodes

        total_rewards = np.empty(n_episodes)


        solved_consecutively = 0

        for i in range(n_episodes):
            state = self.env.reset()
            done = False

            score = 0
            while not done:
                action, q_value = self.get_action(state, epsilon)
                next_state, reward, done, info = self.env.step(action)
                self.memory.add(Transition(state, action, reward, next_state, done))
                score += reward

                state = next_state

                if self.memory.size > self.batch_size:
                    loss = self.train()
                    if i % self.config.target_update_every_step == 0:
                        self.update_target()

            if epsilon > epsilon_final:
                epsilon = max(epsilon_final, epsilon - eps_drop)

            total_rewards[i] = score
            avg_rewards = total_rewards[max(0, i - 100):(i + 1)].mean()

            with self.writer.as_default():
                with tf.name_scope('Performance'):
                    tf.summary.scalar('episode reward', score, step=i)
                    tf.summary.scalar('running avg reward(100)', avg_rewards, step=i)
                    tf.summary.scalar('loss', 0 if loss is None else loss, step=i)
                    tf.summary.histogram('Weights', self.main_net.weights[0], step=i)

            # Specific four mountain car
            if done and score == 500:
                solved_consecutively += 1
            else:
                solved_consecutively = 0

            if solved_consecutively >= 50:
                print(f'Successfully SOLVED {solved_consecutively} times!')
                break

            if i % self.config.log_every_episode == 0:
                print("episode:", i, "/", self.config.n_episodes, "episode reward:", score,"avg reward (last 100):",
                      avg_rewards, "eps:", epsilon, "Learning rate (10e3):",
                      (self.optimizer._decayed_lr(tf.float32).numpy() * 1000),
                      "Consecutively solved:", solved_consecutively)

        plot_learning_curve(self.name + '.png', {'rewards': total_rewards})
        self.save()

    # Summary writing routines

    def write_summaries(self):
        with tf.name_scope('Layer 1'):
            with tf.name_scope('W'):
                mean = tf.reduce_mean(W)
                tf.summary.scalar('mean', mean)
            stddev = tf.sqrt(tf.reduce_mean(tf.square(W - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.histogram('histogram', var)



    def save(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = os.path.join(self.model_dir, current_time + '.h5')
        self.main_net.save(path)

    def load(self, modelh5):
        assert NotImplementedError
        self.main_net = tf.keras.models.load_model(modelh5)

if __name__ == '__main__':
    import gym
    env = gym.make('CartPole-v1')
    dqn = DQNAgent(env, 'SimpleDQN', training=True, config=DefaultConfig())
    dqn.run()
