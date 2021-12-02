# Task Classifier

This module is for selecting the unit task to be performed currently to reach the goal state.

The structure of the module is as follows.

figure

As input to the neural network, images of the current state and the goal state are used.

The algorithm learns through SAC-discrete, and CQL is applied because offline reinforcement learning is used.

Supervised learning and learning of DQN are included for comparison with other algorithms.
