import numpy as np
from collections import namedtuple, deque
import torch 
import config

device = config.device

class SumTree():
    """
    This SumTree code is modified version and the original code is from: 
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    Store the data with its priority in tree and data frameworks.
    """

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.data_pointer = 0
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------parent nodes-------------][-------leaves to record priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)  # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, data):
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data  # update data_frame
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:    # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:     # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1         # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:       # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx] or self.tree[cr_idx] == 0.0:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_priority(self):
        return self.tree[0]  # the root
    
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.data)


class PrioritisedExpReplay():  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py

    and was further adopted to work smoothly with Pytorch and named tuples
    """
    epsilon = config.PER_epsilon  # small amount to avoid zero priority
    alpha = config.PER_alpha  # [0~1] convert the importance of TD error to priority
    beta = config.PER_beta  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = config.PER_beta_increment_per_sampling
    abs_err_upper = config.PER_abs_err_upper  # clipped abs error

    def __init__(self, capacity, batch_size, seed):
        self.tree = SumTree(capacity)
        self.batch_size = batch_size
        self.seed = np.random.seed(seed)

        #type definition for the storage objects - makes extraction easier
        self.experience_type = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def store(self, experience):
        
        state, action, reward, next_state, done = experience
        transition = self.experience_type(state, action, reward, next_state, done)
        
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0.0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)   # set the max p for new p

    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """
    def sample(self):
        # Create a sample array that will contains the minibatch
        memory_b = []
        
        n = self.batch_size
        
        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)
        
        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n       # priority segment
    
        # Here we increasing the beta param each time we sample a new minibatch
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        
        # Calculating the max_weight
        p_min = max(self.epsilon, np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority)
        max_weight = (p_min * n) ** (-self.beta)
        
        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            start, end = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(start, end)
            
            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)
            
            #P(j)
            sampling_probabilities = priority / self.tree.total_priority
            
            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            
            #if i ==7:
            #    print(f"{n} {sampling_probabilities} {self.beta} {max_weight} {p_min}")
            
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.beta)/ max_weight
                                   
            b_idx[i] = index

            experience = data

            memory_b.append(experience)

        experiences_batch = self._extract_tuples(memory_b)
        
        return b_idx, experiences_batch, b_ISWeights
    
    def _extract_tuples(self, experiences_batch):
                
        states = torch.from_numpy(np.vstack([e.state for e in experiences_batch if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences_batch if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences_batch if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences_batch if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences_batch if e is not None]).astype(np.uint8)).float().to(device)
  
        return states, actions, rewards, next_states, dones

    def batch_update(self, tree_idx, abs_errors):
        """
        Batch update is used to recalculate priorities in the sumtree
        """
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)
            
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.tree)
