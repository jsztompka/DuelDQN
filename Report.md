## Report

## Agent is using Duel DQN Algorithm. 

This algorithm has a unique model implementation where the final output consists of 2 separate layers calculating different functions. 
First (output) layer calculates a normal value function , the second (output) layer calculates the Advantage function and the model returns:
Q = V + A - mean(A) 

The implemenation in this repo follows architecture from the paper and it uses weighted Huber-Loss loss function to adjust model I’ve found it adds a little bit of stability to it

On average my best score was between 17-18 points

Config also includes all parameters I found to work best.

If you decide to run training and change parameters here is a list of things which did not work for me:

Bigger networks (128/256/512/1024 - no added value to training)
Bigger UPDATE_STEP - didn’t observe any improvement in training by reducing or increasing
LR - that seems to be the biggest factor combined with optimiser
Optimizer I’ve trained following the same training regime with RMSProp but never got as good results as with Adam
Epochs - if you don’t get great results by 1,000-1500 episode training much further does little to improve and usually introduces just noise to the weights. I went as far as 5,000 and never got any higher score than at 1,000. If you have a better luck do let me know.

## Model 

Model consists of 4 layers, I've found that using 64 nodes per layer works best, details of the model are in the table below. 


|        Layer (type)   |           Output Shape   |      Param #|
| --- | --- | --- | 
|            Linear-1         |      [-1, 64, 64]    |       2,432|
|            Linear-2         |      [-1, 64, 64]    |       4,160|
|            Linear-3         |       [-1, 64, 4]    |         260|
|            Linear-4         |       [-1, 64, 4]    |         260|


Total params: 7,112
Trainable params: 7,112
Non-trainable params: 0

Input size (MB): 0.01
Forward/backward pass size (MB): 0.07
Params size (MB): 0.03
Estimated Total Size (MB): 0.10

## Training chart: 
![](/Training.png)

## Video of the trained agent:
[![Click to watch on youtube](https://img.youtube.com/vi/SRBDl_yjLBM/0.jpg)](https://youtu.be/SRBDl_yjLBM)

## Parameters used (please see config.py): 
### Agent / network specific

BUFFER_SIZE = int(5e4)  - replay buffer size
BATCH_SIZE = 64         - minibatch size
GAMMA = 0.98            - discount factor
TAU = 1e-2              - for soft update of target parameters
LR = 1e-3               - learning rate
UPDATE_EVERY = 4        - how often to update the network

### Prioritised Experience Replay params

Epsilon / alpha - kept the values from the paper , decreased beta and beta_increment slightly
PER_epsilon = 0.001  - small amount to avoid zero priority
PER_alpha = 0.6  - [0~1] convert the importance of TD error to priority
PER_beta = 0.4   - importance-sampling, from initial value increasing to 1
PER_beta_increment_per_sampling = 0.001  - the rate of importance sampling increase
PER_abs_err_upper = 1.  - clipped abs error
