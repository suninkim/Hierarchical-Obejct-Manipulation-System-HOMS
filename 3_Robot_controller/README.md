# Robot Controller

This module is for performing unit tasks with a robot.

The structure of the module is as follows.

figure

As input to the neural network, the image of the current state, the robot state, and the type of task are used.

The algorithm learns through SAC-continuous, and CQL and BC are applied because offline reinforcement learning is used.

At this time, CQL and BC can be selected and applied, and TD3 can be used instead of SAC.
