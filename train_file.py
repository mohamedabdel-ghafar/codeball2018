from ml import LearningAgent, ReplayBuffer
from Env import CodeBallEnv
from Env.marl_codeball_env import count_actions
from numpy import random as np_rand, reshape, array, mean as np_mean, zeros, vstack, repeat
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
                    game_repr = my_env.extract_features([s])
                    action_index_list = sess.run(l_a.main_q.predict, feed_dict={l_a.main_q.input: game_repr})[0]
                new_state, reward, done = my_env.step_discrete(action_index_list)
                ep_buffer.add([[s, action_index_list, my_env.process_state(new_state), reward, done]], flatten=False)
                total_steps += 1
                if total_steps > pre_train_steps:
                    if e > endE:
                        e -= step_drop
                    if total_steps % update_freq == 0:
                        train_batch = array(ep_buffer.sample(batch_size))
                        curr_s_n = train_batch[:, 0]
                        action_n = train_batch[:, 1]
                        next_s_n = train_batch[:, 2]
                        rew_n = train_batch[:, 3]
                        done_n = train_batch[:, 4]
                        curr_s_n = my_env.extract_features(curr_s_n)
                        next_s_n = my_env.extract_features(next_s_n)
                        actions_next_n, q_scores_next = sess.run([l_a.main_q.predict, l_a.target_q.q_out],
                                                                 feed_dict={l_a.main_q.input: next_s_n,
                                                                            l_a.target_q.input: next_s_n})
                        q_sp_a = []
                        for j in range(batch_size):
                            q_sp_a.append(q_scores_next[j][range(team_size), actions_next_n[j]])
                        q_sp_a = array(q_sp_a)
                        end_mul = reshape(1 - done_n, [-1, 1])
                        to_add = y*end_mul*q_sp_a
                        rew_n = vstack(rew_n)
                        q_target = rew_n + to_add
                        action_n = vstack(action_n)
                        sess.run(l_a.main_q.update,
                                 feed_dict={
                                     l_a.main_q.input: curr_s_n,
                                     l_a.main_q.target_q: q_target,
                                     l_a.main_q.action: action_n,
                                 })
                        for op in target_ops:
                            sess.run(op)
                rAll += sum(reward)
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
            if i % 50 == 0:
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
        while s is not None:
            s = env.process_state(s)
            game_repr = env.extract_features([s])
            if i % 10 == 0:
                action_index_list = sess.run(l_a.main_q.predict, feed_dict={l_a.main_q.input: game_repr})[0]
                prev_ac = action_index_list
            i += 1
            print(prev_ac)
            s, _, _ = env.step_discrete(prev_ac)
        env.kill_process()


if __name__ == "__main__":
    train(5, save_dir="./saves", save_rate=5, load_last=False)
    # test_model(4)

