from ml import LearningAgent, ReplayBuffer
from Env import CodeBallEnv
from Env.marl_codeball_env import count_actions
from numpy import random as np_rand, reshape, array, mean as np_mean
import tensorflow as tf
import os
from ml.ml_model import NUM_ACTIONS


# TODO: reward per robot
# TODO: try instead of ball position to train a neural network to predict where the ball will be
# TODO: add dribble action and re-train, improve intercept and kick actions
# TODO: take into account bumping off the wall,
# TODO: add score and time as inputs to the DQN
# TODO: add no-op action
# TODO: instead of returning an action we could return a tuple(action, time to execute)
#       1) while t for this action > 0 -> perform same action type and t--
#       2) once t reaches 0, choose new action according to e-greedy policy


# How many experiences to use for each training step.
batch_size = 4
# How often to perform a training step.
update_freq = 10
# Discount factor on the target Q-values
y = .99
# Starting chance of random action
startE = 1
# Final chance of random action
endE = 0.1
# How many steps of training to reduce startE to endE.
annealing_steps = 10000000
# How many episodes of game environment to train network with.
num_episodes = 10000
# How many steps of random actions before training begins.
pre_train_steps = 10000
# The max allowed length of our episode.
max_epLength = 100000
# The path to save our model to.
path = "./dqn"
# mix percentage between target network current values and primary network values
tau = 0.001


def expand_state(my_team: list, other_team: list, ball_state):
    ball_states = [ball_state for _ in range(len(my_team))]
    team_states = []
    c_robs = []
    for i in range(len(my_team)):
        tmp = my_team
        c_r = tmp.pop(i)
        tmp.extend(other_team)
        c_robs.append(c_r)
        team_states.append(tmp)
    return team_states, c_robs, ball_states


def update_target_graph(tfVars, tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx, var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder


def update_target(op_holder, sess):
    for op in op_holder:
        sess.run(op)


def reshape_samples(samples, team_size):
    return reshape(samples, [len(samples), 2*team_size + 1, 6])


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
    my_buffer = ReplayBuffer()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        sess.run(init)
        if load_last:
            print("loading model,,,")
            ckpt = tf.train.get_checkpoint_state(save_dir)
            st_i = ckpt.model_checkpoint_path
            st_i = os.path.split(st_i)[-1]
            st_i = st_i.split(".")[0].split("-")[1]
            i = int(st_i) + 1
            print("loaded model: ", ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.model_checkpoint_path)
        for i in range(num_episodes):
            ep_buffer = ReplayBuffer()
            s = my_env.reset()
            j = 0
            done = False
            rAll = 0
            while j < max_epLength:
                j += 1
                if np_rand.rand(1) < e or total_steps < pre_train_steps:
                    action_index_list = np_rand.random_integers(0, NUM_ACTIONS-1, team_size)
                else:
                    s = my_env.process_state(s)
                    game_repr = CodeBallEnv.extract_features(s)
                    my_team = game_repr[:team_size]
                    other_team = game_repr[team_size: 2*team_size]
                    ball_state = game_repr[-1]
                    team_states, curr_robs, ball_states = expand_state(my_team, other_team, ball_state)
                    action_index_list = sess.run(l_a.main_q.predict, feed_dict={l_a.main_q.input: team_states,
                                                                                l_a.main_q.ball_state: ball_states,
                                                                                l_a.main_q.curr_rob_state: curr_robs})
                new_state, reward, done = my_env.step_discrete(action_index_list)
                ep_buffer.add([(s, action_index_list, my_env.process_state(new_state), reward, done)], flatten=False)
                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= step_drop
                    if total_steps % update_freq == 0:
                        train_batch = ep_buffer.sample(batch_size)
                        for counter in range(batch_size):
                            curr_s, actions, next_s, r, d,  = train_batch[counter]
                            # now we have to predict actions of next_s using main_q, but estimate Q(s',a) using target_q
                            my_team = next_s[:team_size]
                            other_team = next_s[team_size, 2*team_size]
                            ball_state = next_s[2*team_size]
                            team_states, curr_robs, ball_states = expand_state(my_team, other_team, ball_state)
                            actions, q_scores_next = sess.run([l_a.main_q.predict, l_a.target_q.q_out],
                                                              feed_dict={
                                                                  l_a.main_q.input: team_states,
                                                                  l_a.main_q.curr_rob_state: curr_robs,
                                                                  l_a.main_q.ball_state: ball_states,
                                                                  l_a.target_q.input: team_states,
                                                                  l_a.target_q.curr_rob_state: curr_robs,
                                                                  l_a.target_q.ball_state: ball_states
                                                              })
                            q_sp_a = q_scores_next[range(team_size), actions]
                            rew_rep = array([r for _ in range(team_size)])
                            end_mul = array([1 - int(d) for _ in range(team_size)])
                            q_target = rew_rep + y * end_mul * q_sp_a
                            my_team = s[:team_size]
                            other_team = s[team_size, 2 * team_size]
                            ball_state = s[2 * team_size]
                            team_states, curr_robs, ball_states = expand_state(my_team, other_team, ball_state)
                            sess.run(l_a.main_q.update, feed_dict={
                                l_a.main_q.input: team_states,
                                l_a.main_q.curr_rob_state: curr_robs,
                                l_a.main_q.ball_state: ball_states,
                                l_a.main_q.action: action_index_list,
                                l_a.main_q.target_q: q_target
                            })
                            for op in target_ops:
                                sess.run(op)
                rAll += reward
                s = new_state
                if done or j == max_epLength:
                    print(j)
                    print("scores: {} {}".format(my_env.curr_me_score, my_env.curr_ad_score))
                    print("ep done")
                    break
            my_buffer.add(ep_buffer.buffer, flatten=False)
            jList.append(j)
            rList.append(rAll)
            if i % save_rate == 0:
                saver.save(sess, save_dir + '/model-' + str(i) + '.ckpt')
                print("Saved Model")
                print("count actions: ", list(count_actions))
            if len(rList) % 10 == 0:
                print(total_steps, np_mean(rList[-10:]), e)
        saver.save(sess, save_dir + '/model-' + str(i) + '.ckpt')


def export_model(model_num=None, export_path=""):
    if model_num is None:
        ckpt = tf.train.get_checkpoint_state(os.path.join("", "saves"))
        model_path = ckpt.model_checkpoint_path
    else:
        model_path = os.path.join("saves", "model-{}.ckpt".format(model_num))
    with tf.Session as sess:
        saver = tf.train.Saver()
        l_a = LearningAgent(2)
        saver.restore(sess, model_path)
        tf.io.write_graph(sess.graph, "graph.pbtxt")


def test_model(team_size):
    ckpt = tf.train.get_checkpoint_state(os.path.join("", "saves"))
    model_path = ckpt.model_checkpoint_path
    saver = tf.train.Saver()
    l_a = LearningAgent(team_size)
    env = CodeBallEnv(team_size)
    s = env.reset()
    with tf.Session() as sess:
        while s is not None:
            game_repr = CodeBallEnv.extract_features(s)
            my_team = game_repr[:team_size]
            other_team = game_repr[team_size: 2 * team_size]
            ball_state = game_repr[-1]
            team_states, curr_robs, ball_states = expand_state(my_team, other_team, ball_state)
            action_index_list = sess.run(l_a.main_q.predict, feed_dict={l_a.main_q.input: team_states,
                                                                        l_a.main_q.ball_state: ball_states,
                                                                        l_a.main_q.curr_rob_state: curr_robs})
            env.step_discrete(action_index_list)


if __name__ == "__main__":
    # train(4, save_dir="./saves", save_rate=5, load_last=False)
    export_model(595)

