import utility
from config import *
import numpy as np
import argparse
from DDPG import DDPG
from utils import create_directory
import visualize
import math

parser = argparse.ArgumentParser("DDPG parameters")
parser.add_argument('--max_episodes', type=int, default=10000)
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/DDPG/')
args = parser.parse_args()


def get_reward(state,state_,flag):
    r = 0
    dist = 0
    dist_ = 0
    for i in range(teamA_num):
        dist_ += np.linalg.norm(state_[2 * i : 2 * i + 2])
        dist += np.linalg.norm(state[2 * i : 2 * i + 2])

    if np.linalg.norm(dist) > np.linalg.norm(dist_):
        r += 1
    if np.linalg.norm(dist) < np.linalg.norm(dist_):
        r -= 1
    if np.linalg.norm(state[-2:0]) > np.linalg.norm(state_[-2:0]):
        r += 3
    if np.linalg.norm(state[-2:0]) < np.linalg.norm(state[-2:0]):
        r -= 3
    if flag == 1:
        r += 5*flag

    return(r)



def main():
    env = utility.field(teamA_num, teamB_num, field_width, field_length)
    agentA = DDPG(alpha=0.0001, beta=0.001, state_dim=2*(teamA_num+teamB_num+1),
                 action_dim=2*teamA_num, actor_fc1_dim=512, actor_fc2_dim=256, actor_fc3_dim=128,
                 critic_fc1_dim=512, critic_fc2_dim=256, critic_fc3_dim=128, ckpt_dir=args.checkpoint_dir + '2v1' + '/',
                 batch_size=64)
    create_directory(args.checkpoint_dir+ '2v1' + '/',
                     sub_paths=['Actor', 'Target_actor', 'Critic', 'Target_critic'])
    

    reward_history = []
    avg_reward_history = []
    for episode in range(args.max_episodes):
        flag= 0
        total_reward = 0
        state = env.derive_pos()
        k = 0
        ok = False
        while True:
            env.reset()
            ok = env.collide()
            if ok == True:
                break
        while flag == 0 and k < 2000:
            state = env.derive_pos()
            action = agentA.choose_action(state,train=True)
            action_ = np.hstack([action,np.array([0,0])])
            env.set_vel(action_)
            bug = env.detect_player()
            if bug:
                break
            flag = env.set_coord()
            state_ = env.derive_pos()
            reward = get_reward(state,state_,flag)
            
            agentA.remember(state, action[0], reward,state_,flag)
            agentA.learn()
            
            
            env.detect_player()

            total_reward += reward
            # if((episode + 1)%100 == 0):
            visualize.draw(env.derive_state())
            k = k + 1

        reward_history.append(total_reward)
        avg_reward = np.mean(reward_history[-100:])
        avg_reward_history.append(avg_reward)
        print('Ep: {0} Flag:{1} Reward: {2:f} '.format(episode+1, flag, total_reward))

        if (episode + 1) % 200 == 0:
            agentA.save_models(episode+1)

    # episodes = [i+1 for i in range(args.max_episodes)]
    # plot_learning_curve(episodes, avg_reward_history, title='AvgReward',
    #                     ylabel='reward', figure_file=args.figure_file)


if __name__ == '__main__':
    main()