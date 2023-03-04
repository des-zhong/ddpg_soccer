import utility
from config import *
import numpy as np
import argparse
from DDPG import DDPG
from utils import create_directory
import visualize

parser = argparse.ArgumentParser("DDPG parameters")
parser.add_argument('--max_episodes', type=int, default=200)
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/DDPG/')
args = parser.parse_args()


def get_pos_reward(state, state_):
    reward = []

    for i in range(teamA_num):
        player = state[2 * i: 2 * i + 2]
        gate = state[-2:]
        player_ = state_[2 * i: 2 * i + 2]
        gate_ = state_[-2:]
        # cos = (player_[0] * gate_[0] + player_[1] * gate_[1]) / np.linalg.norm(player_) / np.linalg.norm(gate_)
        # r = 5 * cos
        r = 0.2 * (np.linalg.norm(player) - np.linalg.norm(player_))
        # r += 2 * (-np.linalg.norm(gate_) + np.linalg.norm(gate))
        reward.append(r)
    for i in range(teamB_num):
        r = 1
        reward.append(r)

    return reward[0]


def main():
    env = utility.field(teamA_num, teamB_num, field_width, field_length)
    agentA = DDPG(alpha=actor_lr, beta=critic_lr, state_dim=2 * (teamA_num + teamB_num + 1),
                  action_dim=2 * teamA_num, actor_fc1_dim=fc1_dim, actor_fc2_dim=fc2_dim, actor_fc3_dim=fc3_dim,
                  critic_fc1_dim=fc1_dim, critic_fc2_dim=fc2_dim, critic_fc3_dim=fc3_dim,
                  ckpt_dir=args.checkpoint_dir + 'test' + '/',
                  batch_size=64)
    create_directory(args.checkpoint_dir + 'test' + '/',
                     sub_paths=['Actor', 'Target_actor', 'Critic', 'Target_critic'])
    # agentA.load_models(0, './checkpoints/' + 'DDPG' + '/test/')
    for episode in range(args.max_episodes):
        flag = 0
        S = []
        A = []
        S_ = []
        F = []
        R = []
        k = 0
        while True:
            env.reset()
            ok = env.collide()
            if ok == True:
                break
        N = 2000
        while flag == 0 and k < 2000:
            state = env.derive_pos()
            S.append(state)
            action = []
            action.append(agentA.choose_action(state, train=True))
            action_ = np.array(action).flatten()
            A.append(action_)
            command = np.array([action[0], np.array([0, 0])]).flatten()
            env.set_vel(command)
            bug, kick = env.detect_player()
            if bug:
                flag = -1
                break
            flag = env.set_coord()
            state_ = env.derive_pos()
            R.append(get_pos_reward(state, state_) + kick * 3)
            S_.append(state_)
            F.append(flag)
            if kick == 1 and N > k:
                print('kick')
                N = k
            k = k + 1
        N = min(N, k)
        if flag == 1:
            for i in range(N):
                agentA.remember(S[i], A[i], R[i] + 20, S_[i], F[i])
                agentA.learn()
        if flag == 0:
            for i in range(N):
                agentA.remember(S[i], A[i], R[i] - 5, S_[i], F[i])
                agentA.learn()
        if flag == -1:
            for i in range(N):
                agentA.remember(S[i], A[i], R[i] - 5, S_[i], F[i])
                agentA.learn()

        print('Ep: {0} Flag:{1} '.format(episode + 1, flag))

        if (episode + 1) % 20 == 0:
            agentA.save_models(episode + 1)

    # episodes = [i+1 for i in range(args.max_episodes)]
    # plot_learning_curve(episodes, avg_reward_history, title='AvgReward',
    #                     ylabel='reward', figure_file=args.figure_file)


if __name__ == '__main__':
    main()
