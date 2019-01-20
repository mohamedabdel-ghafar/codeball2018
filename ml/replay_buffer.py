import random
from numpy import reshape


class ReplayBuffer(object):
    def __init__(self, size=50000):
        self.buffer = []
        self.max_len = size

    def add(self, samples: list, flatten=True):
        # if number of experience tuples > max size the leave only max size exp tuples
        if flatten:
            samples = ReplayBuffer.flatten_samples(samples)
        if len(samples) > self.max_len:
            samples = samples[:self.max_len]
        if len(samples) + len(self.buffer) > self.max_len:
            self.buffer[0:len(samples) + len(self.buffer) - self.max_len] = []
        self.buffer.extend(samples)

    def sample(self, num_samples):
        return random.sample(self.buffer, num_samples)

    @staticmethod
    def flatten_samples(samples_list):
        return reshape(samples_list, [len(samples_list), -1])
