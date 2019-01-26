import sys
from time import sleep
from model import *
from MyStrategy import MyStrategy
from RemoteProcessClient import RemoteProcessClient


class Runner:
    PORT1 = 31008

    def read_rules_wrapper(self):
        self.rules = self.remote_process_client.read_rules()
        print(self.rules)
        # self.remote_process_client2.read_rules()
        # print("out of read rules")
        return self.rules

    def read_game_wrapper(self):
        self.curr_game = self.remote_process_client.read_game()
        # self.remote_process_client2.read_game()
        return self.curr_game

    @staticmethod
    def add_to_p(val):
        # Runner.PORT2 += val
        Runner.PORT1 += val

    def __init__(self):
        if sys.argv.__len__() == 4:
            self.remote_process_client = RemoteProcessClient(
                sys.argv[1], int(sys.argv[2]))
            self.token = sys.argv[3]
        else:
            self.remote_process_client = RemoteProcessClient(
                "127.0.0.1", self.PORT1)
            print("in __init__ runner")
            self.token = "0000000000000000"
            self.remote_process_client.write_token(self.token)
        self.indx_to_id = {}
        self.id_to_indx = {}
        self.man_marks = {}
        print("reading rules")
        self.read_rules_wrapper()
        self.curr_game = None
        me_c = 0
        print("reading game")
        self.read_game_wrapper()
        s_game = self.curr_game
        print(s_game)
        robots_list = s_game.robots
        my_robots = list()
        adv_robots = list()
        # print("goin int init runner loop1")
        teammate_q = list()
        self.next_friend = {}
        for robot_c in robots_list:
            if robot_c.is_teammate:
                self.indx_to_id[me_c] = robot_c.id
                self.id_to_indx[robot_c.id] = me_c
                me_c += 1
                my_robots.append(robot_c)
                if len(teammate_q) == 0:
                    teammate_q.append(self.id_to_indx[robot_c.id])
                else:
                    self.next_friend[self.id_to_indx[robot_c.id]] = teammate_q.pop()
            else:
                adv_robots.append(robot_c)
        for robot_c in my_robots:
            self.man_marks[robot_c.id] = adv_robots.pop().id
        print(len(self.next_friend.items()))

    def run2(self, actions: dict):
        actions_me = {}
        for r_indx, action in actions.items():
            r_id = self.indx_to_id[r_indx]
            actions_me[r_id] = action
        self.remote_process_client.write(actions_me, "")
        curr_game = self.read_game_wrapper()
        return curr_game

    def run(self):
        strategy = MyStrategy()

        self.remote_process_client.write_token(self.token)
        rules = self.remote_process_client.read_rules()

        while True:
            game = self.remote_process_client.read_game()
            if game is None:
                break

            actions = {}

            for robot in game.robots:
                if robot.is_teammate:
                    action = Action()
                    strategy.act(robot, rules, game, action)
                    actions[robot.id] = action

            self.remote_process_client.write(actions, strategy.custom_rendering())


# Runner().run()
