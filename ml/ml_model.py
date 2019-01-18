import tensorflow as tf
from tensorflow.nn import dynamic_rnn
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.layers import fully_connected
LSTM_NUM_UNITS = 256
NUM_ACTIONS = 9
STATE_SIZE = 6


class LearningAgent(object):

    def __init__(self, team_size):
        self.main_q = QNetwork(team_size)
        self.target_q = QNetwork(team_size)


class QNetwork(object):
    def __init__(self, team_size):
        # placeholders to be input during train/test time
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, 2 * team_size, STATE_SIZE])
        self.action = tf.placeholder(dtype=tf.int32, shape=[None])
        self.curr_rob_state = tf.placeholder(dtype=tf.float32, shape=[None, 6])
        self.ball_state = tf.placeholder(dtype=tf.float32, shape=[None, 6])
        # slicing
        self.my_team, self.other_team = tf.split(self.input, 2, axis=1)
        my_team_pos = tf.slice(self.my_team, [0, 0, 0], [tf.shape(self.input)[0], team_size, 3])
        other_team_pos = tf.slice(self.other_team, [0, 0, 0], [tf.shape(self.input)[0], team_size, 3])
        my_team_v = tf.slice(self.my_team, [0, 0, 3], [tf.shape(self.input)[0], team_size, 3])
        other_team_v = tf.slice(self.other_team, [0, 0, 3], [tf.shape(self.input)[0], team_size, 3])
        # network
        pos_lstm = LSTMCell(num_units=LSTM_NUM_UNITS, activation=tf.nn.relu, input_shape=[None, team_size, 3])
        vel_lstm = LSTMCell(num_units=LSTM_NUM_UNITS, activation=tf.nn.relu, input_shape=[None, team_size, 3])
        my_vel_repr = dynamic_rnn(vel_lstm, my_team_v)
        opp_vel_repr = dynamic_rnn(vel_lstm, other_team_v)
        my_pos_repr = dynamic_rnn(pos_lstm, my_team_pos)
        opp_pos_repr = dynamic_rnn(pos_lstm, other_team_pos)
        fc_in = tf.concat([self.curr_rob_state, self.ball_state, my_vel_repr,
                           opp_vel_repr, my_pos_repr, opp_pos_repr, self.action], axis=1)
        fc_out = fully_connected(fc_in, 512, reuse=True, scope="fc1_layer")
        classify = fully_connected(fc_out, NUM_ACTIONS, reuse=True, scope="classifiy_fc")
        one_hot_action = tf.one_hot(self.action, NUM_ACTIONS, axis=1)
        self.q = tf.reduce_sum(tf.multiply(classify, one_hot_action), axis = 1)
        self.target_q = tf.placeholder(shape=[None, NUM_ACTIONS])
        self.predict = tf.argmax(classify, 1)
        self.loss = tf.reduce_mean(tf.square(self.q - self.target_q))
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.update = self.trainer.minimize(self.loss)

