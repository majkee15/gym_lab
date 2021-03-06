# dqn policy
import os
import numpy as np
import tensorflow as tf
import datetime

from gym.spaces import Box, Discrete

from policies.memory import Transition, ReplayMemory, PrioritizedReplayMemory
# from policies.prioritized_memory import PrioritizedReplayBuffer

from policies.base import Policy, BaseModelMixin, TrainConfig
from utils.misc import plot_learning_curve
from utils.annealing_schedule import AnnealingSchedule

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class DefaultConfig(TrainConfig):
    n_episodes = 250
    warmup_episodes = 150

    # fixed learning rate
    learning_rate = 0.001
    end_learning_rate = 0.001
    # decaying learning rate
    # learning_rate = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=learning_rate,
    #                                                               decay_steps=n_episodes*10000,
    #                                                               end_learning_rate=end_learning_rate, power=1.0)
    learning_rate_schedule = AnnealingSchedule(learning_rate, end_learning_rate, n_episodes)
    gamma = 1.0
    epsilon = 1.0
    epsilon_final = 0.01
    epsilon_schedule = AnnealingSchedule(epsilon, epsilon_final, warmup_episodes)
    target_update_every_step = 25
    log_every_episode = 10

    # Memory setting
    batch_normalization = False
    batch_size = 128
    memory_size = 100000
    # PER setting
    prioritized_memory_replay = True
    replay_alpha = 0.3
    replay_beta = 0.4
    replay_beta_final = 1.0
    beta_schedule = AnnealingSchedule(replay_beta, replay_beta_final, n_episodes, inverse=True)
    prior_eps = 1e-6


class DQNAgent(Policy, BaseModelMixin):

    def __init__(self, env, name, config=None, training=True, layers=(128, 128, 128)):

        Policy.__init__(self, env, name, training=training)
        BaseModelMixin.__init__(self, name)

        self.env = env
        self.config = config
        self.batch_size = self.config.batch_size

        if config.prioritized_memory_replay:
            self.memory = PrioritizedReplayMemory(alpha=config.replay_alpha, capacity=config.memory_size)
        else:
            self.memory = ReplayMemory(capacity=self.config.memory_size)

        self.layers = layers

        # Optimizer
        self.global_lr = tf.Variable(self.config.learning_rate_schedule.current_p, trainable=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.global_lr) #, clipnorm=100)
        # Huber Loss
        self.h = tf.keras.losses.Huber(delta=10.0, reduction=tf.keras.losses.Reduction.NONE)


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

        # Global step - epoch - number of trainings elapsed
        self.global_step = 0

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
        batch = self.memory.sample(self.batch_size, beta=self.config.beta_schedule.current_p)
        states = batch['s']
        actions = batch['a']
        rewards = batch['r']
        next_states = batch['s_next']
        dones = batch['done']

        if self.config.prioritized_memory_replay:
            idx = batch['indices']
            weights = batch['weights']

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


            # the next action should be selected using the online net

            target_value = tf.reduce_sum(tf.one_hot(next_action, self.act_size) * target_q, axis=1)
            target_value = (1 - dones) * self.config.gamma * target_value + rewards

            main_q = self.main_net.predict_on_batch(states)
            main_value = tf.reduce_sum(tf.one_hot(actions, self.act_size) * main_q, axis=1)

            td_error = target_value - main_value
            element_wise_loss = tf.square(td_error) * 0.5
            # element_wise_loss = self.h(target_value, main_value)
            if self.config.prioritized_memory_replay:
                error = tf.reduce_mean(element_wise_loss * weights)
            else:
                error = tf.reduce_mean(element_wise_loss)


        # update per priorities
        if self.config.prioritized_memory_replay:
            self.memory.update_priorities(idx, np.abs(td_error.numpy()) + self.config.prior_eps)
        dqn_grads = tape.gradient(error, dqn_variable)
        self.optimizer.apply_gradients(zip(dqn_grads, dqn_variable))

        # Logging
        self.global_step += 1
        with self.writer.as_default():
            with tf.name_scope('Network'):
                tf.summary.histogram('Weights', self.main_net.weights[0], step=self.global_step)
                tf.summary.histogram('Gradients', dqn_grads[0], step=self.global_step)
                tf.summary.histogram('Predictions', main_value, step=self.global_step)
                tf.summary.histogram('Target', target_value, step=self.global_step)
                tf.summary.histogram('TD error', td_error, step=self.global_step)
                tf.summary.histogram('Elementwise Loss', element_wise_loss, step=self.global_step)
                tf.summary.scalar('Loss', tf.reduce_mean(element_wise_loss), step=self.global_step)

        return tf.reduce_mean(element_wise_loss)

    def run(self):

        n_episodes = self.config.n_episodes
        loss = None

        total_rewards = np.empty(n_episodes)


        solved_consecutively = 0

        for i in range(n_episodes):
            state = self.env.reset()
            done = False

            score = 0
            while not done:
                action, q_value = self.get_action(state, self.config.epsilon_schedule.current_p)
                next_state, reward, done, info = self.env.step(action)
                self.memory.add(Transition(state, action, reward, next_state, done))
                score += reward

                state = next_state

                if self.memory.size > self.batch_size:
                    loss = self.train()
                    if i % self.config.target_update_every_step == 0:
                        self.update_target()

            total_rewards[i] = score
            avg_rewards = total_rewards[max(0, i - 100):(i + 1)].mean()

            self.config.epsilon_schedule.anneal()
            self.config.beta_schedule.anneal()
            self.global_lr.assign(self.config.learning_rate_schedule.anneal())

            with self.writer.as_default():
                with tf.name_scope('Performance'):
                    tf.summary.scalar('episode reward', score, step=i)
                    tf.summary.scalar('running avg reward(100)', avg_rewards, step=i)

                if self.config.prioritized_memory_replay:
                    with tf.name_scope('Schedules'):
                        tf.summary.scalar('Beta', self.config.beta_schedule.current_p, step=i)
                        tf.summary.scalar('Epsilon', self.config.epsilon_schedule.current_p, step=i)
                        tf.summary.scalar('Learning rate', self.optimizer._decayed_lr(tf.float32).numpy(), step=i)


            # Specific four mountain car
            if done and score == 500:
                solved_consecutively += 1
            else:
                solved_consecutively = 0

            if solved_consecutively >= 50:
                print(f'Successfully SOLVED {solved_consecutively} times!')
                break

            if i % self.config.log_every_episode == 0:
                print("episode:", i, "/", self.config.n_episodes, "episode reward:", score, "avg reward (last 100):",
                      avg_rewards, "eps:",  self.config.epsilon_schedule.current_p, "Learning rate (10e-3):",
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
