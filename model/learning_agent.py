
from keras.models import Model, Sequential
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam
import numpy as np

from rl.processors import WhiteningNormalizerProcessor
from rl.agents import DDPGAgent
from rl.memory import  SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

nb_actions = 1
# 10 vectors each of length 3 (positions and velocities of robots and the ball )
# plus 4 vectors of length 4 each vector consists of touch + touch normal vector, one for each robot
observation_size = 3 * 10 + 4 * 4
# action size is for each robot we need 4 attributs 1 is the jump speed, 3 for target vel.
action_size = 2*4
L1_SIZE = 400
L2_SIZE = 300

ENV_NAME = "code_ball_contest"


class MujocoProcessor(WhiteningNormalizerProcessor):
    def process_action(self, action):
        return np.clip(action, -1., 1.)


class LearningAgent(object):
    def __init__(self):
        self.observation_input = Input(shape=(1, observation_size), name="obs_in")
        self.actor_model = Sequential()
        self.actor_model.add(Flatten(input_shape=(1, observation_size)))
        self.actor_model.add(Dense(L1_SIZE, input_dim=observation_size, name="actor_d_1"))
        self.actor_model.add(Activation('relu'))
        self.actor_model.add(Dense(L2_SIZE, name="actor_d_2"))
        self.actor_model.add(Activation('relu'))
        self.actor_model.add(Dense(action_size, name="actor_d_3"))
        self.actor_model.add(Activation('tanh'))
        self.action_input = Input(shape=(action_size, ), name="action_in")
        flattened_obs = Flatten()(self.observation_input)
        q_net = Dense(L1_SIZE)(flattened_obs)
        q_net = Activation('relu')(q_net)
        q_net = Concatenate()([q_net, self.action_input])
        q_net = Dense(L2_SIZE)(q_net)
        q_net = Activation('relu')(q_net)
        q_net = Dense(1)(q_net)
        q_net = Activation('linear')(q_net)
        self.critic_model = Model(inputs=[self.action_input, self.observation_input], outputs=q_net)
        self.memory = SequentialMemory(limit=100000, window_length=1)
        random_process = OrnsteinUhlenbeckProcess(size=action_size, theta=.15, mu=0., sigma=.1)
        self.agent = DDPGAgent(nb_actions=action_size, actor=self.actor_model,
                               critic=self.critic_model, critic_action_input=self.action_input,
                               memory=self.memory, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,
                               random_process=random_process, gamma=.99, target_model_update=1e-3,
                               processor=MujocoProcessor())

    def compile_agent(self):
        self.agent.compile([Adam(lr=1e-4), Adam(lr=1e-3)], metrics=['mae'])

    def train_agent(self, env):
        self.agent.fit(env=env, nb_steps=1000000, visualize=False, verbose=1)
        self.agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

    @staticmethod
    def load_agent(save_path="./Weights"):
        l_a = LearningAgent()
        l_a.agent.load_weights(save_path)

    def test_agent(self, env):
        self.agent.test(env, nb_episodes=5, visualize=True, nb_max_episode_steps=200)