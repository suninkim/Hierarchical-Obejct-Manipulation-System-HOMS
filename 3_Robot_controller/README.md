# Robot Controller

This module is for performing unit tasks with a robot.

The structure of the module is as follows.

![RC_structure](https://user-images.githubusercontent.com/50347012/144417215-1d4d0a09-84f3-4001-bae1-6a9595486b2b.png)

As input to the neural network, the image of the current state, the robot state, and the type of task are used.

The algorithm learns through SAC-continuous, and CQL and BC are applied because offline reinforcement learning is used.

At this time, CQL and BC can be selected and applied, and TD3 can be used instead of SAC.
