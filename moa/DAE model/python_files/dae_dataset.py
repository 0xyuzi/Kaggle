import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class MoASwapDataset:
    def __init__(self, features, noise):
        self.features = features
        self.noise = noise
        
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
        
        sample = self.features[idx, :].copy()
        sample = self.swap_sample(sample)
        
        dct = {
            'x' : torch.tensor(sample, dtype=torch.float),
            'y' : torch.tensor(self.features[idx, :], dtype=torch.float)            
        }
        return dct
    
    def swap_sample(self,sample):
            #print(sample.shape)
            num_samples = self.features.shape[0]
            num_features = self.features.shape[1]
            if len(sample.shape) == 2:
                batch_size = sample.shape[0]
                random_row = np.random.randint(0, num_samples, size=batch_size)
                for i in range(batch_size):
                    random_col = np.random.rand(num_features) < self.noise
                    #print(random_col)
                    sample[i, random_col] = self.features[random_row[i], random_col]
            else:
                batch_size = 1
          
                random_row = np.random.randint(0, num_samples, size=batch_size)
               
            
                random_col = np.random.rand(num_features) < self.noise
                #print(random_col)
                #print(random_col)
       
                sample[ random_col] = self.features[random_row, random_col]
                
            return sample

class MoADataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)            
        }
        return dct
    

    
class TestDataset:
    def __init__(self, features):
        self.features = features
        
    def __len__(self):
        return (self.features.shape[0])
    
    def __getitem__(self, idx):
        dct = {
            'x' : torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct
    