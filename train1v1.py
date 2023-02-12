import utility
from config import *
import numpy as np
import argparse
from DDPG import DDPG
from utils import create_directory
import visualize

parser = argparse.ArgumentParser("DDPG parameters")
parser.add_argument('--max_episodes', type=int, default=20)
parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/DDPG/')
args = parser.parse_args()


# def get_reward(state, state_, action, flag):
#     reward = []
#     for i in range(teamA_num):
#         player = state[4 * i: 4 * i + 2]
#         soccer = state[4 * (teamA_num + teamB_num):4 * (teamA_num + teamB_num) + 2]
#         player_ = state_[4 * i: 4 * i + 2]
#         gate = np.array([field_width / 2, 0])
#         soccer_ = state_[4 * (teamA_num + teamB_num):4 * (teamA_num + teamB_num) + 2]
#         r = np.linalg.norm(player - soccer) - np.linalg.norm(player_ - soccer_)
#         # r -= np.linalg.norm(soccer_ - gate)
#         # r = -np.linalg.norm(action)
#         reward.append(r)
#     for i in range(teamB_num):
#         r = 1
#         reward.append(r)
#
#     return (reward)


def get_pos_reward(state, state_, action, flag):
    reward = []

    for i in range(teamA_num):
        player = state[2 * i: 2 * i + 2]
        gate = state[-2:]
        player_ = state_[2 * i: 2 * i + 2]
        gate_ = state_[-2:]
        # r = np.linalg.norm(player) - np.linalg.norm(player_)
        # r += 3 * (np.linalg.norm(gate) - np.linalg.norm(gate_))
        cos = (player_[0] * gate_[0] + player_[1] * gate_[1]) / np.linalg.norm(player_) / np.linalg.norm(gate_)
        r = 8*cos
        r = +np.linalg.norm(player) - np.linalg.norm(player_)
        r += 2 * (-np.linalg.norm(gate_) + np.linalg.norm(gate))
        reward.append(r)
    for i in range(teamB_num):
        r = 1
        reward.append(r)

    return (reward)


def main():
    env = utility.field(teamA_num, teamB_num, field_width, field_length)
    agentA = DDPG(alpha=actor_lr, beta=critic_lr, state_dim=2 * (teamA_num + teamB_num + 1),
                  action_dim=2 * teamA_num, actor_fc1_dim=fc1_dim, actor_fc2_dim=fc2_dim, actor_fc3_dim=fc3_dim,
                  critic_fc1_dim=fc1_dim, critic_fc2_dim=fc2_dim, critic_fc3_dim=fc3_dim,
                  ckpt_dir=args.checkpoint_dir + 'test' + '/',
                  batch_size=64)
    create_directory(args.checkpoint_dir + 'test' + '/',
                     sub_paths=['Actor', 'Target_actor', 'Critic', 'Target_critic'])

    reward_history = []
    avg_reward_history = []
    for episode in range(args.max_episodes):
        flag = 0
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
            kick = 0
            state = env.derive_pos()
            action = [agentA.choose_action(state, train=True)]
            # action.append(100*(np.random.random(2 * teamB_num) - np.ones(2 * teamB_num) * 0.5))

            action_ = np.array(action).flatten()
            # print(action_)
            flag = env.run_step(action_)
            state_ = env.derive_pos()
            reward = get_pos_reward(state, state_, action_, flag)

            agentA.remember(state, action[0], reward[0], state_, flag)
            agentA.learn()

            env.detect_player()

            total_reward += np.array(reward)
            # if((episode + 1)%100 == 0):
            visualize.draw(env.derive_state())
            k = k + 1

        reward_history.append(total_reward)
        avg_reward = np.mean(reward_history[-100:])
        avg_reward_history.append(avg_reward)
        print(episode, reward)
        # print('Ep: {0} Flag:{1} Reward: {2:f} {3:f} '.format(episode+1, flag, total_reward[0],total_reward[1]))
    #
    agentA.save_models(0)

    # episodes = [i+1 for i in range(args.max_episodes)]
    # plot_learning_curve(episodes, avg_reward_history, title='AvgReward',
    #                     ylabel='reward', figure_file=args.figure_file)


if __name__ == '__main__':
    main()
