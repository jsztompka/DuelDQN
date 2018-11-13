#Duel Double DQN implementation 

This project is my sample implementation of Duel Double DQN algorithm described in detail:
https://arxiv.org/pdf/1511.06581.pdf

The environment used to present the algorithm is Banas from Unity-ML
You don’t have to build the environment yourself the prebuilt one included in the project will work fine - please note it’s only compatible with Unity-ML 0.4.0b NOT the current newest version. I don’t have access to the source of the environment as it was prebuilt by Udacity. 

#Installation: 
Please run pip install . in order to ensure you got all dependencies needed

To start up the project:
python -m train.py 

All hyper-paramters are in: 
config.py 

It includes PLAY_ONLY argument which decides whether to start Agent with pre-trained weights or spend a few hours and train it from scratch :) 

#Model: 
I’m using PyTorch built NN with 37 inputs (env enforced) -> 64 - > 64 -> 4* 
The network is split into Value and Advantage functions as per Duel network architecture paper

#Agent: 
Follows architecture from the paper and it uses weighted Huber-Loss loss function to adjust model 
I’ve found it adds a little bit of stability to it

On average my best score was between 17-18 points 

Config also includes all parameters I found to work best. 

If you decide to run training and change parameters here is a list of things which didn’t work for me: 
- Bigger networks (128/256/512/1024 - no added value to training) 
- Bigger UPDATE_STEP - didn’t observe any improvement in training by reducing or increasing 
- LR - that seems to be the biggest factor combined with optimiser 
- Optimizer I’ve trained following the same training regime with RMSProp but never got as good results as with Adam 
- Epochs - if you don’t get great results by 1,000-1500 episode training much further does little to improve and usually introduces just noise to the weights. I went as far as 5,000 and never got any higher score than at 1,000. If you have a better luck do let me know. 

#Future improvements: 
The main problem with RL in general seems to be always the noise. If you find a way to reduce it you will get a better results. 

Noisy Nets - introducing this special layer into the network might help reducing the noise 
Rainbow - at the time of writing this state of art algorithm promising much better results 

