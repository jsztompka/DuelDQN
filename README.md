# Duel Double DQN implementation 

This project is my sample implementation of Duel Double DQN algorithm described in detail:
https://arxiv.org/pdf/1511.06581.pdf

The environment used to present the algorithm is Banas from Unity-ML
You don’t have to build the environment yourself the prebuilt one included in the project will work fine - please note it’s only compatible with Unity-ML 0.4.0b NOT the current newest version. I don’t have access to the source of the environment as it was prebuilt by Udacity. 

## Video of the trained agent:
[![Click to watch on youtube](https://img.youtube.com/vi/SRBDl_yjLBM/0.jpg)](https://youtu.be/SRBDl_yjLBM)

## Installation: 
Please run pip install . in order to ensure you got all dependencies needed

To start up the project:
python -m train.py 

All hyper-paramters are in: 
config.py 

It includes PLAY_ONLY argument which decides whether to start Agent with pre-trained weights or spend a few hours and train it from scratch :) 

More details on the project can be found in:  
[Report](/Report.md)



