import DDPG
from config import *
import numpy as np

agentA = DDPG.DDPG(alpha=0.0001, beta=0.001, state_dim=2 * (teamA_num + teamB_num + 1),
                   action_dim=2 * teamA_num, actor_fc1_dim=512, actor_fc2_dim=256, actor_fc3_dim=128,
                   critic_fc1_dim=512, critic_fc2_dim=256, critic_fc3_dim=128,
                   ckpt_dir='./checkpoints/shooting_02' + '/',
                   batch_size=64)
agentA.load_models(240, './checkpoints/' + 'shooting_02' + '/')


def rel_state(state):
    L = teamA_num + teamB_num
    rel_state = np.zeros(state.shape)
    for i in range(L):
        rel_state[2 * i] = state[-2] - state[2 * i]
        rel_state[2 * i + 1] = state[-1] - state[2 * i + 1]
    rel_state[-2] = field_width / 2 - state[-2]
    rel_state[-1] = - state[-1]
    return rel_state


def gen_waypoint(state):
    vel = agentA.choose_action(rel_state(state), False)
    return state[:-2] + vel * time_step


if __name__ == "__main__":
    state = np.array([705.46959907, 218.45933058, 1074.79913256, -107.5407487]).reshape(4, )
    print(state)
    print(gen_waypoint(state))
