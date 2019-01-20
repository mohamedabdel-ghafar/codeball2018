import tensorflow as tf
from tensorflow.nn import dynamic_rnn
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.layers import fully_connected
LSTM_NUM_UNITS = 256
NUM_ACTIONS = 7
STATE_SIZE = 6


class LearningAgent(object):

    def __init__(self, team_size):
        self.main_q = QNetwork(team_size)
        self.target_q = QNetwork(team_size)


class QNetwork(object):
    def __init__(self, team_size):
        # placeholders to be input during train/test time
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, 2 * team_size - 1, STATE_SIZE])
        self.action = tf.placeholder(dtype=tf.int32, shape=[None])
        self.curr_rob_state = tf.placeholder(dtype=tf.float32, shape=[None, STATE_SIZE])
        self.ball_state = tf.placeholder(dtype=tf.float32, shape=[None, STATE_SIZE])
        # slicing
        self.my_team, self.other_team = tf.split(self.input, [team_size - 1, team_size], axis=1)
        my_team_pos, my_team_v = tf.split(self.my_team, [3, 3], axis=2)
        other_team_pos, other_team_v = tf.split(self.other_team, [3, 3], axis=2)
        # network
        pos_lstm = LSTMCell(num_units=LSTM_NUM_UNITS, activation=tf.nn.relu, input_shape=[None, None, 3],
                            dtype=tf.float32)
        vel_lstm = LSTMCell(num_units=LSTM_NUM_UNITS, activation=tf.nn.relu, input_shape=[None, None, 3],
                            dtype=tf.float32)
        _, my_vel_repr = dynamic_rnn(vel_lstm, my_team_v, dtype=tf.float32, scope="my_vel")
        _, opp_vel_repr = dynamic_rnn(vel_lstm, other_team_v, dtype=tf.float32, scope="opp_vel")
        _, my_pos_repr = dynamic_rnn(pos_lstm, my_team_pos, dtype=tf.float32, scope="my_pos")
        _, opp_pos_repr = dynamic_rnn(pos_lstm, other_team_pos, dtype=tf.float32, scope="opp_pos")
        my_vel_repr = my_vel_repr.h
        my_pos_repr = my_pos_repr.h
        opp_vel_repr = opp_vel_repr.h
        opp_pos_repr = opp_pos_repr.h

        fc_in = tf.concat([self.curr_rob_state, self.ball_state, my_vel_repr,
                           opp_vel_repr, my_pos_repr, opp_pos_repr,
                           tf.reshape(tf.cast(self.action, tf.float32), [-1, 1])], axis=1)
        fc_out = fully_connected(fc_in, 512)
        classify = fully_connected(fc_out, NUM_ACTIONS)
        one_hot_action = tf.one_hot(self.action, NUM_ACTIONS)
        self.q = tf.reduce_sum(tf.multiply(classify, one_hot_action), axis=1)
        self.target_q = tf.placeholder(shape=[None, NUM_ACTIONS], dtype=tf.float32)
        self.q_out = classify
        self.predict = tf.argmax(self.q_out, 1)
        self.loss = tf.reduce_mean(tf.square(self.q - self.target_q))
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.update = self.trainer.minimize(self.loss)

