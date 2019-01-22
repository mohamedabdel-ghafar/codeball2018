from model import Action, Robot, Game, Rules
from os import path
from tensorflow import train, Session   
weights_path = path.join("", "model-5.ckpt")
ROBOT_STOP_DISTANCE = 2


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


class MyStrategy:
    l_a = None
    teammates = None
    round_done = None
    actions_to_perform = None
    indx_to_id = None
    id_to_indx = None
    rules = None

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
            return self.through_ball(curr_robot, game.ball)
        elif action_index == 5:
            return self.kick_ball_towards_goal(curr_robot, game.ball)
        elif action_index == 6:
            return self.intercept_closest_enemy(curr_robot, game.robots)

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

    def stand_in_goal(self, curr_robot):
        target_pos_x = (self.rules.arena.goal_width*(self.id_to_indx[curr_robot.id] + 1) /
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
                max_diff_x = max(max_diff_x, enemy_x[indx+1][0] - enemy_x[indx][0])
                begin = enemy_x[indx][0]
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
                    cl_indx = indx
        return self.move_to(curr_robot, robots[cl_indx].x, 0, robots[cl_indx].z)

    def start(self, rules, game):
        self.rules = rules
        i = 0
        self.indx_to_id = {}
        self.id_to_indx = {}
        for robot in game.robots:
            if robot.is_teammate:
                self.indx_to_id[i] = robot.id
                self.id_to_indx[robot.id] = i
                i += 1

    def act(self, me: Robot, rules: Rules, game: Game, action: Action):
        if self.rules is None:
            self.start(rules=rules, game=game)

    def custom_rendering(self):
        return ""
