from config import *
import numpy as np
import control
import visualize


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
        s = dx / d
        c = dy / d
        # if vx1 * c + vy1 * s - vx2 * c - vy2 * s < 0:
        #     return
        # print('kick')
        self.vel = vec2D(vx1 * c ** 2 + vy1 * s * c - vx2 * c ** 2 - vy2 * s * c + vx2 * s ** 2 + -vy2 * s * c + vx1,
                         vx1 * s * c + vy1 * s ** 2 - vx2 * c * s - vy2 * s ** 2 - vx2 * s * c + vy2 * c ** 2 + vy1)

    def crush(self, player):
        d = player.coord.dist(self.coord)
        dx = self.coord.x - player.coord.x
        dy = self.coord.y - player.coord.y
        vy1 = player.vel.y
        vx1 = player.vel.x
        vy2 = self.vel.y
        vx2 = self.vel.x
        s = dx / d
        c = dy / d
        self.vel = vec2D(vx1 * s ** 2 - vy1 * c * s + vx2 * c ** 2 + vy2 * s * c,
                         vx2 * c * s + vy2 * s ** 2 - vx1 * s * c + vy1 * c ** 2)
        player.vel = vec2D(vx2 * s ** 2 - vy2 * c * s + vx1 * c ** 2 + vy1 * s * c,
                           vx1 * c * s + vy1 * s ** 2 - vx2 * s * c + vy2 * c ** 2)

    def vel_fade(self):
        v = self.vel.dist(vec2D(0, 0))
        if v > max_velocity:
            self.vel = self.vel.mul(max_velocity / v)

    def process(self):
        self.coord = self.coord.add(self.vel.mul(time_step))
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

    def reset(self):
        self.teamA = []
        self.teamB = []
        for i in range(self.numA):
            random_coord = random(self.xlimA, self.ylim)
            self.teamA.append(object(random_coord, vec2D(0, 0), radius_player, i))
        for i in range(self.numB):
            random_coord = random(self.xlimB, self.ylim)
            self.teamB.append(object(random_coord, vec2D(0, 0), radius_player, i))
        self.soccer = object(vec2D(0, 0), vec2D(0, 0), radius_soccer)

    def detect_soccer(self):
        if self.soccer.coord.y > self.ylim[1] - self.soccer.radius or self.soccer.coord.y < self.ylim[
            0] + self.soccer.radius or self.soccer.coord.x < -field_width / 2 + self.soccer.radius:
            print('Out of field')
            return -1
        if self.soccer.coord.x > field_width / 2 - self.soccer.radius:
            if self.soccer.coord.y < self.gate_length / 2 - self.soccer.radius and self.soccer.coord.y > -self.gate_length / 2 + self.soccer.radius:
                print('Goal!')
                return 1
            else:
                print('Out of field')
                return -1
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
            if dist_to_ball <= radius_player + radius_soccer:
                self.soccer.strike(self.teamB[i])

    def run_step(self, acc_command):
        state = self.derive_state()
        self.soccer.process()
        for i in range(self.numA):
            delta_v = acc_command[2 * i: 2 * i + 2] * time_step
            self.teamA[i].vel = self.teamA[i].vel.add(vec2D(delta_v[0], delta_v[1]))
            self.teamA[i].process()
            self.teamA[i].vel_fade()
        for i in range(self.numB):
            delta_v = acc_command[2 * self.numA + 2 * i: 2 * self.numA + 2 * i + 2] * time_step
            self.teamB[i].vel = self.teamB[i].vel.add(vec2D(delta_v[0], delta_v[1]))
            self.teamB[i].process()
            self.teamB[i].vel_fade()
        flag = self.detect_soccer()
        state_ = self.derive_state()

        return state_, flag

    def match(self, num):
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
                state = self.derive_state()
                command = control.nn(state)
                state_, reward, flag = self.run_step(command)
                self.detect_player()
                k = k + 1
                visualize.draw(state_)
                if k > max_iter:
                    break
            print('\n')


   