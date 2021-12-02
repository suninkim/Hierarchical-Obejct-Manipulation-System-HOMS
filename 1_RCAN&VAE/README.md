# RCAN & VAE

This module is for learning RCAN and VAE separately from reinforcement learning.

## RCAN

Since HOMS learns from virtual environment data, RCAN is used for simulation-to-real-world tranfer.

In this case, the data is collected by applying domain randomization when offline data is collected.

Data generation is performed in 4_Data_generator.

Learning is performed with the following code.

```p
python RCAN.py
```

## VAE

Since HOMS is an image-based system, it utilizes latent vectors compressed through VAE.

At this time, learning is performed from the dataset of offline reinforcement learning.

Data generation is performed in 4_Data_generator.

Learning is performed with the following code.

```p
python Vae.py
```


