import random


class ReplayBuffer(object):
    def __init__(self, size=50000):
        self.buffer = []
        self.max_len = size

    def add(self, exp_tup_list: list):
        # if number of experience tuples > max size the leave only max size exp tuples
        if len(exp_tup_list) > self.max_len:
            exp_tup_list = exp_tup_list[:self.max_len]
        if len(exp_tup_list) + len(self.buffer) > self.max_len:
            self.buffer[0:len(exp_tup_list) + len(self.buffer) - self.max_len] = []
        self.buffer.extend(exp_tup_list)

    def sample(self, num_samples):
        return random.sample(self.buffer, num_samples)

