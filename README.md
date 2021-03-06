# Hierarchical-Obejct-Manipulation-System-HOMS
The system is intended to perform complex tasks that require procedures.

The system consists of a task classifier and a robot controller, each learning through offline reinforcement learning.

The overall structure of the system is shown in the figure below.

![whole_structure](https://user-images.githubusercontent.com/50347012/144417159-fe7d22e1-331f-4a22-acfa-24452959291d.png)

In this case, the image is compressed using VAE to enable efficient learning.

Because it learns using virtual environment data, it is applied to the real environment using RCAN.

The urdf required for each module and the learned VAE weights can be downloaded from HOMS_urdf and final_latent.pth at the following link.

link: https://drive.google.com/drive/folders/1vb5gxkhQ9PLQNBXB_dwg47qoHiaw5EOL?usp=sharing

A detailed description of each module is in each folder.
