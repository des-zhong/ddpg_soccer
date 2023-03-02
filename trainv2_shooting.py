import utility
from config import *
import numpy as np
import argparse
from DDPG import DDPG
from utils import create_directory
import visualize

parser = argparse.ArgumentParser("DDPG parameters")
parser.add_argument('--max_episodes', type=int, default=5000)
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/')
args = parser.parse_args()



def main():
    env = utility.field(teamA_num, teamB_num, field_width, field_length)
    agentA = DDPG(alpha=0.0001, beta=0.001, state_dim=2 * (teamA_num + teamB_num + 1),
                  action_dim=2 * teamA_num, actor_fc1_dim=512, actor_fc2_dim=256, actor_fc3_dim=128,
                  critic_fc1_dim=512, critic_fc2_dim=256, critic_fc3_dim=128, ckpt_dir=args.checkpoint_dir + 'v2' + '/',
                  batch_size=64)
    create_directory(args.checkpoint_dir + 'v2' + '/',
                     sub_paths=['Actor', 'Target_actor', 'Critic', 'Target_critic'])
    agentA.load_models(240, args.checkpoint_dir + 'shooting_02' + '/')

    reward_history = []
    avg_reward_history = []
    for episode in range(args.max_episodes):
        flag = 0
        S = []
        A = []
        S_ = []
        F = []
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
            S.append(state)
            action = []
            action.append(agentA.choose_action(state, train=True))

            action_ = np.array(action).flatten()
            A.append(action_)
            # print(action_)
            env.set_vel(action_)
            bug = env.detect_player()
            if bug:
                break
            flag = env.set_coord()
            state_ = env.derive_pos()
            S_.append(state_)
            F.append(flag)
            k = k + 1

        if flag == 1:
            for i in range(k):
                agentA.remember(S[i], A[i], 1, S_[i], F[i])
                agentA.learn()
        if flag == 0:
            for i in range(k):
                agentA.remember(S[i], A[i], -1, S_[i], F[i])
                agentA.learn()
        if flag == -1:
            for i in range(k):
                agentA.remember(S[i], A[i], -1, S_[i], F[i])
                agentA.learn()

        print('Ep: {0} Flag:{1} '.format(episode + 1, flag))

        if (episode + 1) % 20 == 0:
            agentA.save_models(episode + 1)

    # episodes = [i+1 for i in range(args.max_episodes)]
    # plot_learning_curve(episodes, avg_reward_history, title='AvgReward',
    #                     ylabel='reward', figure_file=args.figure_file)


if __name__ == '__main__':
    main()