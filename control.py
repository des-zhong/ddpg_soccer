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
    
    # command = (np.random.random(2 * teamA_num + 2 * teamB_num) - np.ones(2 * teamA_num + 2 * teamB_num) * 0.5) * 50
    command = []
    # for i in range(teamA_num):
    command.append(agentA.choose_action(state,train = False))
    # for i in range(teamB_num):
    #     command.append(agentB[i].choose_action(state,train = False))
    command.append(np.random.random(2 * teamB_num) - np.ones(2 * teamB_num) * 0.5)
    command = np.array(command).flatten()


    return 100 * command
    # return command
