import torch 
import numpy as np 

class DataLoader: 
    def __init__(self, X, block_size) -> None:
       self.curr_pos = 0 
       self.X = torch.tensor(X)
       self.examples_index = list(range(0, len(X)-block_size, block_size))
       self.order = np.random.permutation(self.examples_index)
       self.block_size = block_size
       self.stride = 0

    def get_batch(self, batch_size):
        batch_examples = self.order[self.curr_pos:self.curr_pos+batch_size]
        self.curr_pos += len(batch_examples)

        if len(batch_examples) < batch_size: 
            self.curr_pos = 0 
            self.stride = (self.stride + 1) % self.block_size 
            self.examples_index = list(range(self.stride, len(self.X)-self.block_size, self.block_size))
            self.order = np.random.permutation(self.examples_index)
            if  len(batch_examples) == 0:
                batch_examples = self.order[self.curr_pos:self.curr_pos+batch_size]
                self.curr_pos += len(batch_examples)

        x_bat = torch.stack([self.X[idx:idx+self.block_size] for idx in batch_examples], dim=0)
        y_bat = torch.stack([self.X[idx+1:idx+1+self.block_size] for idx in batch_examples], dim=0) 
        return x_bat, y_bat