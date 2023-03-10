from config import *
import numpy as np
import visualize
import DDPG


class vec2D():
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def add(self, vec2):
        return vec2D(self.x + vec2.x, self.y + vec2.y)

    def min(self, vec2):
        return vec2D(self.x - vec2.x, self.y - vec2.y)

    def mul(self, c):
        return vec2D(self.x * c, self.y * c)

    def dist(self, vec2):
        return np.sqrt((self.x - vec2.x) ** 2 + (self.y - vec2.y) ** 2)


def random(xlim, ylim):
    x = np.random.uniform(xlim[0], xlim[1])
    y = np.random.uniform(ylim[0], ylim[1])
    return vec2D(x, y)


class object():
    def __init__(self, coord, vel, radius, index=-1):
        self.coord = coord
        self.vel = vel
        self.radius = radius
        self.index = index

    def strike(self, player):
        d = player.coord.dist(self.coord)
        dx = self.coord.x - player.coord.x
        dy = self.coord.y - player.coord.y
        vy1 = player.vel.y
        vx1 = player.vel.x
        vy2 = self.vel.y
        vx2 = self.vel.x
        s = dy / d
        c = dx / d
        vrx = vx1 - vx2
        vry = vy1 - vy2
        # if vrx * c + vry * s < 0:
        #     return
        self.vel = vec2D(2 * vry * s * c + vrx * (c ** 2 - s ** 2) + vx1,
                         2 * vrx * s * c - vry * (c ** 2 - s ** 2) + vy1)

    def soccer_bounce(self, xbounds):
        coef = 0.9
        if xbounds:
            self.vel.x = -coef*self.vel.x
        else:
            self.vel.y = -coef*self.vel.y

    def crush(self, player):
        coef = -1
        d = player.coord.dist(self.coord)
        dx = self.coord.x - player.coord.x
        dy = self.coord.y - player.coord.y
        vy1 = player.vel.y
        vx1 = player.vel.x
        vy2 = self.vel.y
        vx2 = self.vel.x
        s = dy / d
        c = dx / d
        self.vel = vec2D(vx1 * s ** 2 - vy1 * c * s + vx2 * c ** 2 + vy2 * s * c - coef * d * c,
                         vx2 * c * s + vy2 * s ** 2 - vx1 * s * c + vy1 * c ** 2 - coef * d * s)
        player.vel = vec2D(vx2 * s ** 2 - vy2 * c * s + vx1 * c ** 2 + vy1 * s * c + coef * d * c,
                           vx1 * c * s + vy1 * s ** 2 - vx2 * s * c + vy2 * c ** 2 + coef * d * s)

    def vel_fade(self):
        v = self.vel.dist(vec2D(0, 0))
        if v > max_velocity:
            self.vel = self.vel.mul(max_velocity / v)

    def process(self, fade):

        self.coord = self.coord.add(self.vel.mul(time_step))
        if not fade:
            return
        prev_vel = self.vel
        self.vel = self.vel.min(self.vel.mul(gamma * time_step))
        if self.vel.x * prev_vel.x < 0:
            self.vel.x = 0
        if self.vel.y * prev_vel.y < 0:
            self.vel.y = 0


class field():
    def __init__(self, teamA_num, teamB_num, width, length):
        self.numA = teamA_num
        self.numB = teamB_num
        self.length = length
        self.width = width

        self.gate_length = gate_length
        # self.xlimA = [-width / 2 + radius_player, width / 2 - radius_player]
        self.xlimA = [-width / 2 + radius_player, 0]
        self.xlimB = [0, width / 2 - radius_player]
        self.ylim = [-length / 2 + radius_player, length / 2 - radius_player]
        self.score = [0, 0]
        self.teamA = []
        self.teamB = []
        for i in range(teamA_num):
            random_coord = random(self.xlimA, self.ylim)
            self.teamA.append(object(random_coord, vec2D(0, 0), radius_player, i))
        for i in range(teamB_num):
            random_coord = random(self.xlimB, self.ylim)
            self.teamB.append(object(random_coord, vec2D(0, 0), radius_player, i))
        self.soccer = object(vec2D(0, 0), vec2D(0, 0), radius_soccer)

    def derive_state(self):
        state = []
        for i in range(self.numA):
            state.append(self.teamA[i].coord.x)
            state.append(self.teamA[i].coord.y)
            state.append(self.teamA[i].vel.x)
            state.append(self.teamA[i].vel.y)
        for i in range(self.numB):
            state.append(self.teamB[i].coord.x)
            state.append(self.teamB[i].coord.y)
            state.append(self.teamB[i].vel.x)
            state.append(self.teamB[i].vel.y)
        state.append(self.soccer.coord.x)
        state.append(self.soccer.coord.y)
        state.append(self.soccer.vel.x)
        state.append(self.soccer.vel.y)
        return np.array(state)

    def derive_pos(self):
        state = []
        for i in range(self.numA):
            state.append(-self.teamA[i].coord.x + self.soccer.coord.x)
            state.append(-self.teamA[i].coord.y + self.soccer.coord.y)
        for i in range(self.numB):
            state.append(-self.teamB[i].coord.x + self.soccer.coord.x)
            state.append(-self.teamB[i].coord.y + self.soccer.coord.y)
        state.append(field_width / 2 - self.soccer.coord.x)
        state.append(-self.soccer.coord.y)
        return np.array(state)

    def reset(self):
        self.teamA = []
        self.teamB = []
        for i in range(self.numA):
            random_coord = random(self.xlimA, self.ylim)
            self.teamA.append(object(random_coord, vec2D(0, 0), radius_player, i))
            # self.teamA.append(object(vec2D(-300, 300), vec2D(0, 0), radius_player, i))
        for i in range(self.numB):
            random_coord = vec2D(500, 0)
            self.teamB.append(object(random_coord, vec2D(0, 0), radius_player, i))
        # self.soccer = object(random([-self.width / 4, self.width / 4], [-self.length / 4, self.length / 4]),
        #                      vec2D(0, 0), radius_soccer)
        random_coord = random([-300, 300], [-300, 300])
        self.soccer = object(random_coord, vec2D(0, 0), radius_soccer)
        # self.soccer = object(vec2D(0,0), vec2D(0, 0), radius_soccer)

    def detect_soccer(self):
        if self.soccer.coord.y > self.ylim[1] - self.soccer.radius or self.soccer.coord.y < self.ylim[
            0] + self.soccer.radius:
            self.soccer.soccer_bounce(False)
        if self.soccer.coord.x < -field_width / 2 + self.soccer.radius:
            self.soccer.soccer_bounce(True)
        if self.soccer.coord.x > field_width / 2 - self.soccer.radius:
            if self.soccer.coord.y < self.gate_length / 2 - self.soccer.radius and self.soccer.coord.y > -self.gate_length / 2 + self.soccer.radius:
                print('Goal!')
                return 1
            else:
                self.soccer.soccer_bounce(True)
        return 0

    def collide(self):
        for i in range(self.numA):
            for j in range(i + 1, self.numA):
                dist_to_other = self.teamA[i].coord.dist(self.teamA[j].coord)
                if dist_to_other <= 2 * radius_player:
                    return False
            for j in range(0, self.numB):
                dist_to_other = self.teamA[i].coord.dist(self.teamB[j].coord)
                if dist_to_other <= 2 * radius_player:
                    return False
            dist_to_ball = self.teamA[i].coord.dist(self.soccer.coord)
            if dist_to_ball <= radius_player + radius_soccer:
                return False
        for i in range(self.numB):
            for j in range(i + 1, self.numB):
                dist_to_other = self.teamB[i].coord.dist(self.teamB[j].coord)
                if dist_to_other <= 2 * radius_player:
                    return False
            dist_to_ball = self.teamB[i].coord.dist(self.soccer.coord)
            if dist_to_ball <= radius_player + radius_soccer:
                return False
        return True

    def detect_player(self):
        for i in range(self.numA):
            if abs(self.teamA[i].coord.x) > field_width / 2 - radius_player:
                self.teamA[i].coord.x = (field_width / 2 - radius_player) * self.teamA[i].coord.x / abs(
                    self.teamA[i].coord.x)
            if abs(self.teamA[i].coord.y) > field_length / 2 - radius_player:
                self.teamA[i].coord.y = (field_length / 2 - radius_player) * self.teamA[i].coord.y / abs(
                    self.teamA[i].coord.y)
            for j in range(i + 1, self.numA):
                dist_to_other = self.teamA[i].coord.dist(self.teamA[j].coord)
                if dist_to_other <= 2 * radius_player:
                    self.teamA[i].crush(self.teamA[j])
            for j in range(0, self.numB):
                dist_to_other = self.teamA[i].coord.dist(self.teamB[j].coord)
                if dist_to_other <= 2 * radius_player:
                    self.teamA[i].crush(self.teamB[j])
            dist_to_ball = self.teamA[i].coord.dist(self.soccer.coord)
            if dist_to_ball < radius_player:
                return True
            if dist_to_ball <= radius_player + radius_soccer:
                self.soccer.strike(self.teamA[i])
        for i in range(self.numB):
            if abs(self.teamB[i].coord.x) > field_width / 2 - radius_player:
                self.teamB[i].coord.x = (field_width / 2 - radius_player) * self.teamB[i].coord.x / abs(
                    self.teamB[i].coord.x)
            if abs(self.teamB[i].coord.y) > field_length / 2 - radius_player:
                self.teamB[i].coord.y = (field_length / 2 - radius_player) * self.teamB[i].coord.y / abs(
                    self.teamB[i].coord.y)
            for j in range(i + 1, self.numB):
                dist_to_other = self.teamB[i].coord.dist(self.teamB[j].coord)
                if dist_to_other <= 2 * radius_player:
                    self.teamB[i].crush(self.teamB[j])
            dist_to_ball = self.teamB[i].coord.dist(self.soccer.coord)
            if dist_to_ball < radius_player:
                return True
            if dist_to_ball <= radius_player + radius_soccer:
                self.soccer.strike(self.teamB[i])
        return False

    def set_vel(self, command):
        for i in range(self.numA):
            self.teamA[i].vel = vec2D(command[2 * i], command[2 * i + 1])
        for i in range(self.numB):
            self.teamB[i].vel = vec2D(command[2 * self.numA + 2 * i], command[2 * self.numA + 2 * i + 1])

    def set_coord(self):
        for i in range(self.numA):
            self.teamA[i].process(False)
        for i in range(self.numB):
            self.teamB[i].process(False)
        flag = self.detect_soccer()
        self.soccer.process(True)

        return flag

    def match(self, num):
        agentA = DDPG.DDPG(alpha=0.0001, beta=0.001, state_dim=2 * (teamA_num + teamB_num + 1),
                           action_dim=2 * teamA_num, actor_fc1_dim=512, actor_fc2_dim=256, actor_fc3_dim=128,
                           critic_fc1_dim=512, critic_fc2_dim=256, critic_fc3_dim=128,
                           ckpt_dir='./checkpoints/shooting_02' + '/',
                           batch_size=64)
        agentA.load_models(240, './checkpoints/' + 'shooting_02' + '/')
        for i in range(num):
            print('match ', i, ' begins')
            flag = 0
            ok = False
            while True:
                self.reset()
                ok = self.collide()
                if ok == True:
                    break
            state = self.derive_state()
            # print(state)
            k = 0
            while flag == 0:
                state = self.derive_pos()
                action = agentA.choose_action(state, train=False)
                action_ = np.hstack([action, np.array([0, 0])])
                # action_ = [agentA.choose_action(state, train=False)]
                # action_ = np.array(action_).flatten()
                self.set_vel(action_)
                bug = self.detect_player()
                if bug:
                    break
                flag = self.set_coord()

                k = k + 1
                visualize.draw(self.derive_state())
                if k > max_iter:
                    break
            print('\n')

    def test_collide(self, num):
        for i in range(num):
            print('match ', i, ' begins')
            flag = 0
            ok = False
            while True:
                self.reset()
                ok = self.collide()
                if ok == True:
                    break
            k = 0
            while flag == 0:
                state = self.derive_pos()
                action = state[:-2]
                # action_ = [agentA.choose_action(state, train=False)]
                # action_ = np.array(action_).flatten()
                self.set_vel(action)
                bug = self.detect_player()
                if bug:
                    break
                flag = self.set_coord()
                k = k + 1
                visualize.draw(self.derive_state())
                if k > max_iter:
                    break
            print('\n')
