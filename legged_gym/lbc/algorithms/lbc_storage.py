from collections import deque
import torch


class LbcStorage():
    def __init__(self): #, num_envs, num_steps, image_dim, dm_dim
        self.image_buf = deque()
        self.dm_buf = deque()

    def add_pair(self, image, dm):
        """
            image: envs * H * W 
        """
        self.image_buf.append(image)
        self.dm_buf.append(dm)
    
    def clear(self):
        self.image_buf.clear()
        self.dm_buf.clear()
    
    def image_batch(self):
        return torch.cat(self.image_buf, dim=0)
    
    def dm_batch(self):
        return torch.cat(self.dm_buf, dim=0)