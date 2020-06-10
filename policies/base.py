import os
import logging
import datetime

import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete
from gym.utils import colorize

from utils.misc import Config
from utils.misc import REPO_ROOT, RESOURCE_ROOT


class TrainConfig(Config):
    lr = 0.001
    n_steps = 10000
    warmup_steps = 5000
    batch_size = 64
    log_every_step = 1000

    # give an extra bonus if done; only needed for certain tasks.
    done_reward = None


class Policy:
    def __init__(self, env, name, training=True, deterministic=False):
        self.env = env
        self.training = training
        self.name = name

        if deterministic:
            np.random.seed(1)
            tf.set_random_seed(1)

        # # Logger:
        # print('Getting logger')
        # self.logger = logging.getLogger(name)
        # self.logger.setLevel(os.getenv('LOG_LEVEL', 'INFO'))
        # self.logger.warning('Instantiated class ' + self.__class__.__name__)


    @property
    def act_size(self):
        # number of options of an action; this only makes sense for discrete actions.
        if isinstance(self.env.action_space, Discrete):
            return self.env.action_space.n
        else:
            return None

    @property
    def act_dim(self):
        # dimension of an action; this only makes sense for continuous actions.
        if isinstance(self.env.action_space, Box):
            return list(self.env.action_space.shape)
        else:
            return []

    @property
    def state_dim(self):
        # dimension of a state.
        return list(self.env.observation_space.shape)

    def obs_to_inputs(self, ob):
        return ob.flatten()

    def act(self, state, **kwargs):
        pass

    def build(self):
        pass

    def train(self, *args, **kwargs):
        pass

    def evaluate(self, n_episodes):
        reward_history = []
        reward = 0.

        for i in range(n_episodes):
            ob = self.env.reset()
            done = False
            while not done:
                a = self.act(ob)
                new_ob, r, done, _ = self.env.step(a)
                self.env.render()
                reward += r
                ob = new_ob

            reward_history.append(reward)
            reward = 0.

        print("Avg. reward over {} episodes: {:.4f}".format(n_episodes, np.mean(reward_history)))


class BaseModelMixin:

    def __init__(self, model_name):
        self._saver = None
        self._writer = None
        self.model_name = model_name
        self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    def _get_dir(self, dir_name):
        path = os.path.join(RESOURCE_ROOT, dir_name, self.model_name, self.current_time)
        os.makedirs(path, exist_ok=True)
        return path

    @property
    def log_dir(self):
        return self._get_dir('training_logs')

    @property
    def checkpoint_dir(self):
        return self._get_dir('checkpoints')

    @property
    def model_dir(self):
        return self._get_dir('models')

    @property
    def tb_dir(self):
        # tensorboard
        return self._get_dir('tb_logs')

    @property
    def writer(self):
        if self._writer is None:
            self._writer = tf.summary.create_file_writer(self.tb_dir)
        return self._writer
