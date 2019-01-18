from ml import LearningAgent, ReplayBuffer
from Env import CodeBallEnv
from numpy import random as np_rand
import tensorflow as tf
import os
from ml.ml_model import NUM_ACTIONS

# How many experiences to use for each training step.
batch_size = 32
# How often to perform a training step.
update_freq = 4
# Discount factor on the target Q-values
y = .99
# Starting chance of random action
startE = 1
# Final chance of random action
endE = 0.1
# How many steps of training to reduce startE to endE.
annealing_steps = 10000
# How many episodes of game environment to train network with.
num_episodes = 10000
# How many steps of random actions before training begins.
pre_train_steps = 10000
# The max allowed length of our episode.
max_epLength = 50
# The path to save our model to.
path = "./dqn"
# mix percentage between target network current values and primary network values
tau = 0.001


def update_target_graph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder


def update_target(op_holder, sess):
    for op in op_holder:
        sess.run(op)


def train(team_size, save_rate, save_dir, load_last=False):
    my_env = CodeBallEnv(team_size)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    l_a = LearningAgent(team_size)
    train_vars = tf.trainable_variables()
    target_ops = update_target_graph(train_vars, tau)
    step_drop = (startE - endE) // annealing_steps
    e = startE
    jList = []
    rList = []
    total_steps = 0

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init)
        if load_last:
            print("loading model,,,")
            ckpt = tf.train.get_checkpoint_state(save_dir)
            saver.restore(sess, ckpt.model_checkpoint_path)
        for i in range(num_episodes):
            ep_buffer = ReplayBuffer()
            my_team, other_team, ball_state = my_env.reset()
            j = 0
            while j < max_epLength:
                j += 1
                if np_rand.rand(1) < e or total_steps < pre_train_steps:
                    action_index_list = np_rand.random_integers(0, NUM_ACTIONS, team_size)
                    # next_s, r_list,
                else:
                    team_states = []
                    curr_rob = []
                    ball_states = [ball_state for _ in range(len(list(my_team.values())))]
                    for r_id, r_state in my_team.items():
                        all_states = my_team
                        curr_rob.insert(my_env.process_runner.id_to_indx[r_id], all_states.pop(r_id))
                        team_states.insert(my_env.process_runner.id_to_indx[r_id], all_states.values())
                        team_states[my_env.process_runner.id_to_indx[r_id]].extend(other_team)
                    action_index_list = sess.run(l_a.main_q.predict, feed_dict={l_a.main_q.input: team_states,
                                                                                l_a.main_q.ball_state: ball_states,
                                                                                l_a.main_q.curr_rob_state: curr_rob})





