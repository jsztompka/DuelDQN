import torch

# The network will use predefined weights and will not train if this is set to true
PLAY_ONLY = True
SAVED_WEIGHTS_FILE = 'model-checkpoints/Bananas-solution.pth'

#Check if GPU is available, otherwise use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Agent / network specific
BUFFER_SIZE = int(3e4)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.98             # discount factor
TAU = 1e-2              # for soft update of target parameters
LR = 1e-2               # learning rate
UPDATE_EVERY = 4        # how often to update the network

# Prioritised Experience Replay params
# kept epsilon / alpha same as per paper , decreased beta and beta_increment slightly
PER_epsilon = 0.001  # small amount to avoid zero priority
PER_alpha = 0.6  # [0~1] convert the importance of TD error to priority
PER_beta = 0.4   # importance-sampling, from initial value increasing to 1
PER_beta_increment_per_sampling = 0.001  # the rate of importance sampling increase
PER_abs_err_upper = 1.  # clipped abs error