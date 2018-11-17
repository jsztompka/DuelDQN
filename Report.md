## Report

## Agent is using Duel DQN Algorithm. 

This algorithm has a unique model implementation where the final output consists of 2 separate layers calculating different functions. 
First (output) layer calculates a normal value function , the second (output) layer calculates the Advantage function and the model returns:
Q = V + A - mean(A) 

The implemenation in this repo follows architecture from the paper but it uses weighted Huber-Loss loss function to calculate the cost(loss). In my experiments Huber-Loss had better performance over standard loss functions (this might be due to using PER), it appears to make the training a little more stable as well. 

On average my best score was between 17-18 points

Config also includes all parameters I found to work best.



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
