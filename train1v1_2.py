import utility
from config import *
import numpy as np
import argparse
from DDPG import DDPG
from utils import create_directory
import visualize

parser = argparse.ArgumentParser("DDPG parameters")
parser.add_argument('--max_episodes', type=int, default=20000)
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/DDPG/')
args = parser.parse_args()


def get_reward(state,state_,flag):
    reward = []
    for i in range(teamA_num):
        player = state[4 * i : 4 * i + 2]
        soccer = state[4*(teamA_num+teamB_num):4*(teamA_num+teamB_num)+2]
        player_ = state_[4 * i : 4 * i + 2]
        soccer_ = state_[4*(teamA_num+teamB_num):4*(teamA_num+teamB_num)+2]
        y_ = (soccer_[1]-player_[1])/(soccer_[0]-player_[0])*(field_length/2-player_[0]) + player_[1]
        y = (soccer[1]-player[1])/(soccer[0]-player[0])*(field_length/2-player[0]) + player[1]
        r = np.linalg.norm(player - soccer) - np.linalg.norm(player_ - soccer_) 
        if (y_ < gate_length/2 and y_ > -gate_length/2) and (y > gate_length/2 or y < -gate_length/2):
             r += 1000
        if (y < gate_length/2 and y > -gate_length/2) and (y_ > gate_length/2 or y_ < -gate_length/2):
            r -= 1000
        r += 1000*flag
        reward.append(r)
    for i in range(teamB_num):
        r = 1
        reward.append(r)

    return(reward)



def main():
    env = utility.field(teamA_num, teamB_num, field_width, field_length)
    agentA = DDPG(alpha=0.0001, beta=0.001, state_dim=4*(teamA_num+teamB_num+1),
                 action_dim=2*teamA_num, actor_fc1_dim=512, actor_fc2_dim=256, actor_fc3_dim=128,
                 critic_fc1_dim=512, critic_fc2_dim=256, critic_fc3_dim=128, ckpt_dir=args.checkpoint_dir + '02' + '/',
                 batch_size=256)
    create_directory(args.checkpoint_dir+ '02' + '/',
                     sub_paths=['Actor', 'Target_actor', 'Critic', 'Target_critic'])
    

    reward_history = []
    avg_reward_history = []
    for episode in range(args.max_episodes):
        flag= 0
        total_reward = 0
        state = env.derive_state()
        k = 0
        ok = False
        while True:
            env.reset()
            ok = env.collide()
            if ok == True:
                break
        while flag == 0 and k < 1000:
            kick = 0
            state = env.derive_state()
            action = []
            action.append(agentA.choose_action(state,train=True))
            action.append(100*(np.random.random(2 * teamB_num) - np.ones(2 * teamB_num) * 0.5))

            action_ = np.array(action).flatten()
            #print(action_)
            state_, flag = env.run_step(action_)
            reward = get_reward(state,state_,flag)
            if kick == 0 and np.linalg.norm(state[4*(teamA_num+teamB_num):4*(teamA_num+teamB_num)+2]- state_[4*(teamA_num+teamB_num):4*(teamA_num+teamB_num)+2]) > 1 :
                reward[0] += 1000

            agentA.remember(state, action[0], reward[0],state_,flag)
            agentA.learn()
            
            
            env.detect_player()

            total_reward += np.array(reward)
            # if((episode + 1)%100 == 0):
            #visualize.draw(state_)
            k = k + 1

        reward_history.append(total_reward)
        avg_reward = np.mean(reward_history[-100:])
        avg_reward_history.append(avg_reward)
        print('Ep: {0} Flag:{1} Reward: {2:f} {3:f} '.format(episode+1, flag, total_reward[0],total_reward[1]))

        if (episode + 1) % 200 == 0:
            agentA.save_models(episode+1)

    # episodes = [i+1 for i in range(args.max_episodes)]
    # plot_learning_curve(episodes, avg_reward_history, title='AvgReward',
    #                     ylabel='reward', figure_file=args.figure_file)


if __name__ == '__main__':
    main()