import argparse
import datetime
import gym
import numpy as np
import itertools
from collections import deque

from env.env import PandaEnv
import time

parser = argparse.ArgumentParser(description='HOMS data generator')
parser.add_argument('--env-name', default="Data-Gen",
                    help='Data generator for HOMS')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--epi', type=int, default=100, metavar='N',
                    help='Load episode (default: 100)')
parser.add_argument('--rollout', type=int, default=1, metavar='N',
                    help='Num of rollout (default: 1)')
parser.add_argument('--gui', action="store_true")

args = parser.parse_args()

# Environment
env = PandaEnv(args, gui=args.gui)
env.seed(args.seed)

total_numsteps = 0
rollout = []

save_epi = 0
saved = True
num_epi_per_rollout = 100
total_save_rollout = 20
non_success = 0
non_success_num = 20
saved_num = 0

for i_episode in itertools.count(1):

    observations = []
    actions = []
    rewards = []
    terminals = []
    possible_action = []    
    
    s = np.array([0,0,0,0,0,0])
    s[args.stage-1] = 1
    d = False
    o = env.reset()
    next_o = None

    episode_reward = 0
    episode_steps = 0


    while not d:
        next_o, r, d, a, pos = env.step() # Step

        observations.append(o)
        actions.append(a)
        rewards.append(r)
        terminals.append(d)
        possible_action.append(pos)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += r

        o = next_o

    if episode_steps > 1:
        next_observations = observations[1:]
        next_observations.append(next_o)

        actions = np.array(actions)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 1)
        observations = np.array(observations)
        next_observations = np.array(next_observations)
        possible_action = np.array(possible_action)
        
        epi_dict = dict(
            observations=observations,
            actions=actions,
            rewards=np.array(rewards).reshape(-1,1),
            next_observations=next_observations,
            terminals=np.array(terminals).reshape(-1,1),
            possible_action=possible_action)

        if saved_num < total_save_rollout:
            if d and episode_steps != 1.0:
                rollout.append(epi_dict)
                save_epi += 1
                saved = False
            elif non_success < non_success_num:
                rollout.append(epi_dict)
                save_epi += 1
                saved = False
                non_success += 1    

    # High-level policy data
    if save_epi % num_epi_per_rollout ==0 and not saved:
        save = np.array(rollout)
        np.save(f'rollout/high_level/rollout{int(save_epi / num_epi_per_rollout)}.npy',save)
        print(f'high_level rollout{int(save_epi / num_epi_per_rollout)} is saved')
        saved=True
        non_success = 0
        rollout = []
        saved_num+=1

    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))
   
env.close()

