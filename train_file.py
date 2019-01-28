from ml import LearningAgent, ReplayBuffer
from Env import CodeBallEnv
from Env.marl_codeball_env import count_actions
from numpy import random as np_rand, reshape, array, mean as np_mean, zeros, shape, int32 as np_int32
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
update_freq = 500
# Discount factor on the target Q-values
y = .99
# Starting chance of random action
startE = 1
# Final chance of random action
endE = 0.1
#
total_steps_2 = sum([962623, 962777, 964412, 961574, 964653, 963611, 963644, 963174])
# How many steps of training to reduce startE to endE.
annealing_steps = 10000000
# How many episodes of game environment to train network with.
num_episodes = 10000
# How many steps of random actions before training begins.
pre_train_steps = 100
# The max allowed length of our episode.
max_epLength = 100000
# The path to save our model to.
path = "./dqn"
# mix percentage between target network current values and primary network values
tau = 0.001


def expand_state(my_team: list, other_team: list, ball_state):
    ball_states = []
    team_states = []
    c_robs = []
    for i in range(len(my_team)):
        c_i = my_team[i]
        f_p = my_team[:i].tolist()
        if i < len(my_team) - 1:
            s_p = my_team[i+1:]
            f_p.extend(s_p)
        f_p.extend(other_team)
        c_robs.append(c_i)
        team_states.append(f_p)
        ball_states.append(ball_state)
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
    step_drop = (startE - endE) / annealing_steps
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
                s = my_env.process_state(s)
                if np_rand.rand(1) < e or total_steps < pre_train_steps:
                    action_index_list = np_rand.random_integers(0, NUM_ACTIONS-1, team_size)
                else:
                    game_repr = CodeBallEnv.extract_features(s)
                    my_team = game_repr[:team_size]
                    other_team = game_repr[team_size: 2*team_size]
                    ball_state = game_repr[-1]
                    team_states, curr_robs, ball_states = expand_state(my_team, other_team, ball_state)
                    action_index_list = sess.run(l_a.main_q.predict, feed_dict={l_a.main_q.input: team_states,
                                                                                l_a.main_q.ball_state: ball_states,
                                                                                l_a.main_q.curr_rob_state: curr_robs,
                                                                                l_a.main_q.action:
                                                                                    zeros([team_size], dtype=np_int32)})
                new_state, reward, done = my_env.step_discrete(action_index_list)
                ep_buffer.add([(s, action_index_list, my_env.process_state(new_state), reward, done)], flatten=False)
                total_steps += 1
                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= step_drop
                    if total_steps % update_freq == 0:
                        train_batch = ep_buffer.sample(batch_size)
                        for counter in range(batch_size):
                            curr_s, actions, next_s, r, d,  = train_batch[counter]
                            # now we have to predict actions of next_s using main_q, but estimate Q(s',a) using target_q
                            next_s = CodeBallEnv.extract_features(next_s)
                            my_team = next_s[:team_size]
                            other_team = next_s[team_size: 2*team_size]
                            ball_state = next_s[2*team_size]
                            team_states, curr_robs, ball_states = expand_state(my_team, other_team, ball_state)
                            actions, q_scores_next = sess.run([l_a.main_q.predict, l_a.target_q.q_out],
                                                              feed_dict={
                                                                  l_a.main_q.input: team_states,
                                                                  l_a.main_q.curr_rob_state: curr_robs,
                                                                  l_a.main_q.ball_state: ball_states,
                                                                  l_a.main_q.action: zeros([team_size], dtype=np_int32),
                                                                  l_a.target_q.input: team_states,
                                                                  l_a.target_q.curr_rob_state: curr_robs,
                                                                  l_a.target_q.ball_state: ball_states,
                                                                  l_a.target_q.action:
                                                                      zeros([team_size], dtype=np_int32)
                                                              })
                            q_sp_a = q_scores_next[range(team_size), actions]
                            rew_rep = array([r for _ in range(team_size)])
                            end_mul = array([1 - int(d) for _ in range(team_size)])
                            q_target = rew_rep + y * end_mul * q_sp_a
                            curr_s = CodeBallEnv.extract_features(curr_s)
                            my_team = curr_s[:team_size]
                            other_team = curr_s[team_size: 2 * team_size]
                            ball_state = curr_s[2 * team_size]
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
                print("curr e:", round(e, 4))
            if i % 20 == 0:
                test_model(team_size)
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

    env = CodeBallEnv(team_size, show=True)
    s = env.reset()
    i = 0
    prev_ac = zeros([team_size])
    test_graph = tf.Graph()
    with tf.Session(graph=test_graph) as sess:
        l_a = LearningAgent(team_size)
        saver = tf.train.Saver()
        saver.restore(sess=sess, save_path=model_path)

        with tf.variable_scope("test_graph") as test_grpah:
            while s is not None:
                s = env.process_state(s)
                game_repr = CodeBallEnv.extract_features(s)
                my_team = game_repr[:team_size]
                other_team = game_repr[team_size: 2 * team_size]
                ball_state = game_repr[-1]
                team_states, curr_robs, ball_states = expand_state(my_team, other_team, ball_state)
                if i % 10 == 0:
                    action_index_list = sess.run(l_a.main_q.predict, feed_dict={l_a.main_q.input: team_states,
                                                                                l_a.main_q.ball_state: ball_states,
                                                                                l_a.main_q.curr_rob_state: curr_robs,
                                                                                l_a.main_q.action: zeros([team_size])})
                    prev_ac = action_index_list
                i += 1
                print(prev_ac)
                s, _, _ = env.step_discrete(prev_ac)
            env.kill_process()


if __name__ == "__main__":
    train(5, save_dir="./saves", save_rate=5, load_last=True)
    # test_model(4)

