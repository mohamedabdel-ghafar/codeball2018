from Runner import Runner
from model import Action, Game, Rules
from numpy import dot, sum, sqrt, square, array, zeros
from os.path import join, abspath
import subprocess


EXE_PATH = join("C:\\", "Users", "Mohamed Abdelghaffar", "Downloads", "Compressed", "codeball2018-windows2",
                "codeball2018.exe")

# EXE_PATH = join("home", "mohamed", "Downloads", "codeball2018-linux", "codeball2018")
EXE_PATH = abspath(EXE_PATH)


ACTION_SHAPE = 5
STATE_SIZE = 28

ROBOT_STOP_DISTANCE = 2
# features based on which I evaluate each robot in my team :
# TODO: implement the man marks part , currently man mark is unused except for observations
#   distance between me and ball - distance between man_mark[me] and ball
#   distance between ball and my goal
#   distance between ball and other goal
#   if goal scored against me
#   if goal scored for my team
#   penalize my y coordinate
#  if ball is behind me then distance else 0.5
w_eval = [-0.3, -0.3, 0.3, -2, 2, -0.2, -0.3]


def eval_rob(features):
    return dot(w_eval, features)


def distance_3d(x1, y1, z1, x2, y2, z2):
    return ((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)**0.5


def distance_2d(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


def dot_2d(x1, y1, x2, y2):
    return x1 * x2 + y1 * y2


def length_2d(x1, y1):
    return (x1**2 + y1**2)**0.5


def dot_3d(x1, y1, z1, x2, y2, z2):
    return x1 * x2 + y1 * y2 + z1 * z2


def length_3d(x, y, z):
    return (x**2 + y**2 + z**2) ** 0.5


def sum_3d(x1, y1, z1, x2, y2, z2):
    return x1 + x2, y1 + y2, z1+z2


def diff_3d(x1, y1, z1, x2, y2, z2):
    return x1 - x2, y1 - y2, z1 - z2


class CodeBallEnv(object):
    # n is the number of agents (adv + good)
    n: int
    # list of Space objects defining the state space of each agent
    observation_space: list()
    action_space: list()
    process_runner: Runner
    T_ADD = 1
    curr_me_score: int
    curr_ad_score: int
    rules: Rules
    teammates: list()

    def __init__(self, n):
        self.n = n
        self.observation_space = [Space(True, (STATE_SIZE, )) for _ in range(self.n)]
        self.action_space = [Space(True, (ACTION_SHAPE, )) for _ in range(self.n)]
        self.local_process = None
        self.process_runner = None

    def reset(self):
        self.curr_me_score = 0
        self.curr_ad_score = 0
        if self.local_process is not None:
            self.local_process.kill()
            self.local_process.wait(60*60)
            Runner.add_to_p(self.T_ADD)
            self.T_ADD *= -1

        # self.local_process = subprocess.Popen([EXE_PATH,  "--no-countdown", "--p1", "tcp-{}".format(Runner.PORT1),
        #                                        "--p2", "keyboard"])
        self.local_process = subprocess.Popen([EXE_PATH,   "--no-countdown", "--p1", "tcp-{}".format(Runner.PORT1),
                                              "--p2", "helper"])
        if self.process_runner is not None:
            self.process_runner.remote_process_client.socket.close()
        self.process_runner = Runner()
        self.rules = self.process_runner.rules
        self.teammates = list(filter(lambda rob: rob.is_teammate, self.process_runner.curr_game.robots))
        return CodeBallEnv.process_state(self.process_runner.curr_game)

    def land_to_ground(self):
        action = Action()
        action.target_velocity_x = 0
        action.target_velocity_z = 0
        action.target_velocity_y = -1.0 * self.rules.MAX_ENTITY_SPEED
        return action

    def move_to(self, curr_robot, x, y, z, move_on_ground=True):
        action = Action()
        action.target_velocity_x = (x - curr_robot.x) * self.rules.ROBOT_MAX_GROUND_SPEED
        if move_on_ground:
            action.target_velocity_y = 0
        else:
            action.target_velocity_y = (y - curr_robot.y)*self.rules.ROBOT_MAX_GROUND_SPEED
        action.target_velocity_z = (z - curr_robot.z) * self.rules.ROBOT_MAX_GROUND_SPEED
        return action

    @staticmethod
    def process_state(game: Game):
        my_team = dict()
        other_team = list()
        for c_robot in game.robots:
            rob_state = [c_robot.x, c_robot.y, c_robot.z, c_robot.velocity_x, c_robot.velocity_y, c_robot.velocity_z]
            if c_robot.is_teammate:
                my_team[c_robot.id] = rob_state
            else:
                other_team.append(rob_state)
        ball_state = [game.ball.x, game.ball.y, game.ball.z, game.ball.velocity_x, game.ball.velocity_y,
                      game.ball.velocity_z]

        return my_team, other_team, ball_state

    def stand_in_goal(self, curr_robot):
        target_pos_x = (self.rules.arena.goal_width*(self.process_runner.id_to_indx[curr_robot.id] + 1) /
                        (2*self.rules.team_size)) - self.rules.arena.goal_width/2.0
        target_pos_z = - 0.5 * self.rules.arena.depth + self.rules.arena.bottom_radius
        if not curr_robot.touch:
            return self.land_to_ground()
        return self.move_to(curr_robot, target_pos_x, 0, target_pos_z)

    def pass_ball_to_closest_friend(self, curr_robot, game: Game):
        dist_to_ball = distance_3d(curr_robot.x, curr_robot.y, curr_robot.z, game.ball.x, game.ball.y, game.ball.z)
        if not curr_robot.touch:
            return self.land_to_ground()
        if dist_to_ball > ROBOT_STOP_DISTANCE + game.ball.radius:
            return self.move_to(curr_robot, game.ball.x + self.rules.BALL_RADIUS, 0,
                                game.ball.z + self.rules.BALL_RADIUS)
        min_dist = 10000
        min_indx = -1
        for i, friend in enumerate(self.teammates):
            if friend.id != curr_robot.id:
                dist = distance_2d(friend.x, friend.z, curr_robot.x, curr_robot.z)
                if dist < min_dist:
                    min_dist = dist
                    min_indx = i
        diff_x = game.ball.x - self.teammates[min_indx].x
        diff_z = game.ball.z - self.teammates[min_indx].z
        norm_fac = length_2d(diff_x, diff_z)
        vec_mag = dot_2d(diff_x, diff_z, curr_robot.x, curr_robot.z) / norm_fac
        target_x = vec_mag * diff_x / norm_fac
        target_z = vec_mag * diff_z / norm_fac
        vec2_x = curr_robot.x - game.ball.x
        vec2_z = curr_robot.z - game.ball.z
        if dot_2d(diff_x, diff_z, vec2_x, vec2_z) < ROBOT_STOP_DISTANCE + game.ball.radius:
            # then we can pass
            action = Action()
            action.jump_speed = self.rules.ROBOT_MAX_JUMP_SPEED
            action.target_velocity_x = self.rules.ROBOT_MAX_GROUND_SPEED * (target_x - curr_robot.x)
            action.target_velocity_z = self.rules.ROBOT_MAX_GROUND_SPEED * (target_z - curr_robot.z)
            return action
        else:
            return self.move_to(curr_robot, target_x, 0, target_z)

    def kick_ball_towards_goal(self, curr_rob, ball):
        dist_to_ball = distance_2d(curr_rob.x, curr_rob.z, ball.x, ball.z)
        if dist_to_ball > 2**0.5 * ball.radius + 0.01:
            return self.move_to(curr_rob, ball.x - ball.radius, 0, ball.z - ball.radius)
        else:
            action = Action()
            action.target_velocity_x = 0
            action.target_velocity_z = self.rules.ROBOT_MAX_GROUND_SPEED
            action.jump_speed = self.rules.ROBOT_MAX_JUMP_SPEED
            return action

    def through_ball(self, curr_rob, ball):
        if distance_2d(curr_rob.x, curr_rob.z, ball.x, ball.z) > ball.radius:
            return self.move_to(curr_rob, ball.x, 0, ball.z - ball.radius)
        action = Action()
        action.target_velocity_z = self.rules.ROBOT_MAX_GROUND_SPEED*0.8
        jump = ball.y - ball.radius > 0.05
        if jump:
            action.jump_speed = self.rules.ROBOT_MAX_JUMP_SPEED
        else:
            action.jump_speed = 0.5 * self.rules.ROBOT_MAX_JUMP_SPEED
        return action

    def move_to_empty_space(self, curr_robot, game):

        if game.ball.z > curr_robot.z:
            enemy_x = list(map(lambda rob: (rob.x, rob.z), filter(lambda rb: not rb.is_teammate and rb.z < game.ball.z,
                                                         game.robots)))
        else:
            enemy_x = list(map(lambda rob: (rob.x, rob.z), filter(lambda rb: not rb.is_teammate and rb.z > game.ball.z,
                                                         game.robots)))

        if len(enemy_x) == 0:
            return Action()
        elif len(enemy_x) == 1:
            if enemy_x[0][0] < 0:
                # move right
                target_x = (self.rules.arena.width - enemy_x[0][0]) / 2
            else:
                # move left
                target_x = (enemy_x[0][0] + self.rules.arena.width/2) / 2
            target_z = enemy_x[0][1]
            return self.move_to(curr_robot, target_x, 0, target_z)
        else:
            sorted(enemy_x, key=lambda x: x[0])
            max_diff_x = -1
            begin = self.rules.arena.width / 2
            for indx in range(len(enemy_x) - 1):
                max_diff_x = max(max_diff_x, enemy_x[i+1][0] - enemy_x[i][0])
                begin = enemy_x[i][0]
            if max_diff_x < self.rules.arena.width / 2 - enemy_x[-1][0]:
                begin = enemy_x[-1][0]
                max_diff_x = self.rules.arena.width / 2 - enemy_x[-1][0]
            target_z = sum(list(map(lambda x: x[1], enemy_x))) / len(enemy_x)
            return self.move_to(curr_robot, begin + max_diff_x/2, 0, target_z)

    def intercept_ball(self, curr_rob, ball):
        if ball.y - ball.radius < 0.01:
            return self.move_to(curr_rob, ball.x, 0, ball.z - ball.radius)
        else:
            if distance_2d(curr_rob.x, curr_rob.z, ball.x, ball.z - ball.radius) < 0.5:
                action = Action()
                action.jump_speed = self.rules.ROBOT_MAX_JUMP_SPEED
                return action
            else:
                return self.move_to(curr_rob, ball.x, 0, ball.z - ball.radius)

    def intercept_closest_enemy(self, curr_robot, robots):
        cl_indx = -1
        cl_dist = 1000000
        for indx, rob in enumerate(robots):
            if not rob.is_teammate:
                new_dist = distance_2d(curr_robot.x, curr_robot.z, rob.x, rob.z)
                if new_dist < cl_dist:
                    cl_dist = new_dist
                    cl_indx  = indx
        return self.move_to(curr_robot, robots[cl_indx].x, 0, robots[cl_indx].z)

    @staticmethod
    def jump(jump_mag):
        action = Action()
        action.jump_speed = jump_mag
        return action

    def perform_action(self, curr_robot, game, action_index):
        if action_index == 0:
            return self.stand_in_goal(curr_robot)
        elif action_index == 1:
            return self.pass_ball_to_closest_friend(curr_robot, game)
        elif action_index == 2:
            return self.intercept_ball(curr_robot, game.ball)
        elif action_index == 3:
            return self.move_to_empty_space(curr_robot, game)
        elif action_index == 4:
            return self.through_ball(curr_robot, game)
        elif action_index == 5:
            return self.kick_ball_towards_goal(curr_robot, game.ball)
        elif action_index == 6:
            return self.intercept_closest_enemy(curr_robot, game.robots)
        elif action_index == 7:
            return CodeBallEnv.jump(0.5*game.rules.ROBOT_MAX_JUMP_SPEED)
        else:
            return CodeBallEnv.jump(1*game.rules.ROBOT_MAX_JUMP_SPEED)

    def step_discrete(self, action_n):
        to_write_actions = {}
        for robot_c in self.process_runner.curr_game.robots:
            if robot_c.is_teammate:
                ac_index = self.process_runner.id_to_indx[robot_c.id]
                to_write_actions[robot_c.id] = self.perform_action(robot_c, self.process_runner.curr_game,
                                                                   ac_index)
        self.process_runner.remote_process_client.write(to_write_actions, "")
        new_game_state = self.process_runner.read_game_wrapper()
        my_indx = 1 - int(new_game_state.players[0].me)
        if new_game_state.players[my_indx].score != self.curr_me_score:
            reward = 1
        elif new_game_state.players[1 - my_indx].score != self.curr_ad_score:
            reward = -1
        else:
            reward = -0.01*(self.rules.arena.depth/2 + self.rules.arena.bottom_radius - new_game_state.ball.z)
        new_game_state = CodeBallEnv.process_state(new_game_state)
        return new_game_state, reward

    def step(self, action_n):
        # first we have to send the actions to the process, them wait for new state
        # we also have to return rewards for each robot
        action_ret = dict()
        for r_indx, action_p in action_n.items():
            n_action = Action()
            n_action.target_velocity_x = float(action_p[0])
            n_action.target_velocity_y = float(action_p[1])
            n_action.target_velocity_z = float(action_p[2])
            n_action.jump_speed = float(action_p[3])
            action_ret[r_indx] = n_action
        n_game = self.process_runner.run2(action_ret)
        if n_game is None:
            done_n = [True for _ in range(self.n)]
            r_n = [0 for _ in range(self.n)]
            obs_n = [zeros([STATE_SIZE]) for _ in range(self.n)]
            return obs_n, r_n, done_n, []
        ball_state = (n_game.ball.x, n_game.ball.y, n_game.ball.z, n_game.ball.velocity_x, n_game.ball.velocity_y,
                      n_game.ball.velocity_z)
        id_to_indx = self.process_runner.id_to_indx
        man_marks = self.process_runner.man_marks
        obs_n = list()
        r_n = list()
        info_n = []
        dist_ball_opp = self.process_runner.rules.arena.depth / 2 - n_game.ball.z
        dist_ball_me = n_game.ball.z + self.process_runner.rules.arena.depth / 2.0
        my_indx = 1 - int(n_game.players[0].me)
        ad_obs = {}
        # print("my id to index ", self.process_runner.id_to_indx)
        for robot_c in n_game.robots:
            eval_features = tuple()
            # print("current robot: ", robot_c)
            if robot_c.is_teammate:
                # print("team mate : ", robot_c.id)
                dist_my_rob_ball = sqrt(sum(square([robot_c.x, robot_c.y, robot_c.z]) - square(array(ball_state[:3]))))
                goal_for_me = self.curr_me_score == n_game.players[my_indx].score
                goal_against_me = self.curr_ad_score == n_game.players[1 - my_indx].score
                self.curr_me_score = n_game.players[my_indx].score
                self.curr_ad_score = n_game.players[1 - my_indx].score
                dir_ball_me = max(0, robot_c.z - ball_state[2])
                eval_features += (dist_my_rob_ball, dist_ball_me, dist_ball_opp, goal_against_me, goal_for_me,
                                  robot_c.y, dir_ball_me)
                robot_state = (robot_c.x, robot_c.y, robot_c.z)
                robot_state += (robot_c.velocity_x, robot_c.velocity_y, robot_c.velocity_z)
                robot_state += (robot_c.touch, )
                if robot_c.touch:
                    robot_state += (robot_c.touch_normal_x, robot_c.touch_normal_y, robot_c.touch_normal_z)
                else:
                    robot_state += (-100, -100, -100)
                robot_state += ball_state
                r_n.insert(id_to_indx[robot_c.id], eval_rob(array(eval_features)))
                obs_n.insert(id_to_indx[robot_c.id], robot_state)
            else:
                ad_obs[robot_c.id] = (robot_c.x, robot_c.y, robot_c.z, robot_c.velocity_x, robot_c.velocity_y,
                                      robot_c.velocity_z)
            # print("num robots me", len(obs_n))
            # print("ad robots", list(ad_obs.keys()))
        for my_id, ad_id in man_marks.items():
            to_add_obs = ad_obs[ad_id]
            obs_n[id_to_indx[my_id]] += to_add_obs
        for my_indx, fr_indx in self.process_runner.next_friend.items():
            obs_n[my_indx] += obs_n[fr_indx][:6]
            obs_n[fr_indx] += obs_n[my_indx][:6]
        done_n = [False for _ in range(self.n)]
        return obs_n, r_n, done_n, info_n


class Space(object):
    # true -> continuous, false -> discrete
    type: bool
    shape: tuple

    def __init__(self, type, shape):
        self.type = type
        self.shape = shape


if __name__ == "__main__":
    n_env = CodeBallEnv(2)
    n_env.reset()
    actions = {}
    pass_agent = min(list(map(lambda r: r.id, n_env.teammates)))
    while True:
        i = 0
        n_env.step_discrete([1, 2])
