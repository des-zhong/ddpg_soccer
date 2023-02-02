from config import *
import numpy as np
from DDPG import DDPG
from utils import create_directory


# agentA = []
# agentB = []
# for i in range(teamA_num):
# agentA = DDPG(alpha=0.01, beta=0.01, state_dim=4*(teamA_num+teamB_num+1),
#                  action_dim=4, actor_fc1_dim=256, actor_fc2_dim=128,
#                  critic_fc1_dim=256, critic_fc2_dim=128, ckpt_dir='./Offense/',
#                  batch_size=256)
# agentA.load_models(15800)
# for i in range(teamB_num):
#         agentB.append (DDPG(alpha=0.01, beta=0.01, state_dim=4*(teamA_num+teamB_num+1),
#                  action_dim=2, actor_fc1_dim=256, actor_fc2_dim=128,
#                  critic_fc1_dim=256, critic_fc2_dim=128, ckpt_dir='./checkpoints/DDPG/' + 'B' + str(i) + '/',
#                  batch_size=256))
#         agentB[i].load_models(800)

def nn(state):
    agentA = DDPG(alpha=0.0001, beta=0.001, state_dim=4 * (teamA_num + teamB_num + 1),
                  action_dim=2 * teamA_num, actor_fc1_dim=512, actor_fc2_dim=256, actor_fc3_dim=128,
                  critic_fc1_dim=512, critic_fc2_dim=256, critic_fc3_dim=128,
                  ckpt_dir='./checkpoints/DDPG/' + 'test' + '/', batch_size=64)
    action = [agentA.choose_action(state, train=False)]
    action.append(100 * (np.random.random(2 * teamB_num) - np.ones(2 * teamB_num) * 0.5))
    action = np.array(action).flatten()
    return action
    # return command
