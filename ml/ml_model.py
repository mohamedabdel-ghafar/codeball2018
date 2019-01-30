import tensorflow as tf
from tensorflow.nn import dynamic_rnn
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib.layers import fully_connected, softmax
LSTM_NUM_UNITS = 256
NUM_ACTIONS = 8
STATE_SIZE = 6


class LearningAgent(object):

    def __init__(self, team_size):
        self.main_q = QNetwork(team_size)
        self.target_q = QNetwork(team_size)


class QNetwork(object):
    def __init__(self, team_size):
        # placeholders to be input during train/test time
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, 2 * team_size + 1, STATE_SIZE])
        # slicing
        self.my_team, self.other_team, self.ball_state = tf.split(self.input, [team_size, team_size, 1], axis=1)
        my_team_pos, my_team_v = tf.split(self.my_team, [3, 3], axis=2)
        other_team_pos, other_team_v = tf.split(self.other_team, [3, 3], axis=2)
        ball_pos, ball_v = tf.split(self.ball_state, [3, 3], axis=2)
        # network
        pos_lstm = LSTMCell(num_units=LSTM_NUM_UNITS, input_shape=[None, None, 3],
                            dtype=tf.float32, activation=tf.nn.tanh)
        vel_lstm = LSTMCell(num_units=LSTM_NUM_UNITS, input_shape=[None, None, 3],
                            dtype=tf.float32, activation=tf.nn.tanh)
        _, my_vel_repr = dynamic_rnn(vel_lstm, my_team_v, dtype=tf.float32, scope="my_vel")
        _, opp_vel_repr = dynamic_rnn(vel_lstm, other_team_v, dtype=tf.float32, scope="opp_vel")
        _, my_pos_repr = dynamic_rnn(pos_lstm, my_team_pos, dtype=tf.float32, scope="my_pos")
        _, opp_pos_repr = dynamic_rnn(pos_lstm, other_team_pos, dtype=tf.float32,  scope="opp_pos")
        _, ball_pos_rep = dynamic_rnn(pos_lstm, ball_pos, dtype=tf.float32, scope="ball_pos")
        _, ball_v_rep = dynamic_rnn(vel_lstm, ball_v, dtype=tf.float32, scope="ball_vel")
        my_vel_repr = my_vel_repr.h
        my_pos_repr = my_pos_repr.h
        opp_vel_repr = opp_vel_repr.h
        opp_pos_repr = opp_pos_repr.h
        ball_pos_rep = ball_pos_rep.h
        ball_v_rep = ball_v_rep.h

        # state stream
        fc_in = tf.concat([my_vel_repr, opp_vel_repr, my_pos_repr, opp_pos_repr, ball_pos_rep,
                           ball_v_rep], axis=1)
        fc_out = fully_connected(fc_in, 1024, activation_fn=tf.nn.leaky_relu)
        # v(s)
        v_s = fully_connected(fc_out, 1, activation_fn=tf.nn.leaky_relu)
        # a(s,a)
        a_s_a = fully_connected(fc_out, NUM_ACTIONS, activation_fn=tf.nn.leaky_relu)
        a_s_a = a_s_a - tf.reduce_mean(a_s_a, axis=1, keepdims=True)
        self.action = tf.placeholder(dtype=tf.int32, shape=[None, team_size])
        one_hot_action = tf.one_hot(self.action, depth=NUM_ACTIONS, axis=-1)
        q_l = a_s_a + v_s
        q_l = tf.reshape(q_l, [-1, 1, NUM_ACTIONS])
        self.q_out = q_l
        action_lstm = LSTMCell(num_units=NUM_ACTIONS, input_shape=[None, None, NUM_ACTIONS])
        for i in range(1, team_size):
            _, q_l = dynamic_rnn(action_lstm, self.q_out, dtype=tf.float32, scope="action_"+str(i))
            q_l = q_l.h
            self.q_out = tf.concat([self.q_out, tf.reshape(q_l, [-1, 1, NUM_ACTIONS])], axis=1)

        self.q = tf.reduce_sum(tf.multiply(self.q_out, one_hot_action), axis=2)
        print(self.q)
        self.target_q = tf.placeholder(shape=[None, team_size], dtype=tf.float32)
        self.predict = tf.argmax(self.q_out, 2)
        self.loss = tf.reduce_mean(tf.square(self.q - self.target_q))
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.update = self.trainer.minimize(self.loss)

