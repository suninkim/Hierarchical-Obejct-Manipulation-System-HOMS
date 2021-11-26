import argparse
import datetime
import gym
import numpy as np
import itertools
from collections import deque

from env.rcan import PandaEnv
import time


parser = argparse.ArgumentParser(description='RCAN data generator')
parser.add_argument('--env-name', default="RCAN GEN",
                    help='Data generator for RCAN')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--rollout', type=int, default=1, metavar='N',
                    help='Num of rollout (default: 1)')
parser.add_argument('--gui', action="store_true")

args = parser.parse_args()

# Environment
env = PandaEnv(args, gui=args.gui)
env.seed(args.seed)

total_numsteps = 0

for i_episode in itertools.count(1):
    d = False
    o = env.reset()

    while not d:
        next_o, r, d, a, pos = env.step()
        total_numsteps += 1
    print("Episode: {}, total numsteps: {}".format(i_episode, total_numsteps))
   
env.close()

