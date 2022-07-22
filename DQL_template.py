import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random


class DQLAgent:
    def __init__(self, env):
        # hyper parameters / parameter
        pass

    def build_model(self):
        # neural network for deep Q learning
        pass

    def remember(self, state, action, reward, done):
        # storage
        pass

    def act(self, state):
        # acting
        pass

    def replay(self, batch_size):
        # training
        pass

    def adaptiveEGreedy(self):
        pass

    def targetModelUpdate(self):
        pass


if __name__ == "__main__":
    # initialize env and agent

    episodes = 100
    for e in range(episodes):
        # initialize env
        while True:

            # act

            # step

            # remember

            # update state

            # replay

            # adjust epsilon

            if done:
                break
