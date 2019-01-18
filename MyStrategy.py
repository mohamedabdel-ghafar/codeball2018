import keras
from rl.agents import DDPGAgent
from model import Action, Robot, Game, Rules
from rl.agents import DDPGAgent
weights_path = "./ddpg_code_ball_contest_weights_actor.h5f"


class MyStrategy:
    # is_first_call = True
    # first_id = 0

    def extract_features(self, me, rules, game):
        # if self.is_first_call:
        #     self.first_id = me.id
        #     self.is_first_call = False
        #
        # return tuple()
        return None

    def act(self, me: Robot, rules: Rules, game: Game, action: Action):
        pass

    def custom_rendering(self):
        return ""
