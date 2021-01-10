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


class Model(nn.Module):
    def __init__(self, num_features, num_targets, hidden_size,hidden_size2,drop_rate1=0.2,drop_rate2=0.5,drop_rate3=0.8):
        super(Model, self).__init__()
        self.batch_norm1 = nn.BatchNorm1d(num_features)
        #self.dropout1 = nn.Dropout(drop_rate1)
        self.dense1 = nn.utils.weight_norm(nn.Linear(num_features, hidden_size))
        
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
      #  self.dropout2 = nn.Dropout(drop_rate2)
        self.dense2 = nn.utils.weight_norm(nn.Linear(hidden_size, hidden_size2))
        
        self.batch_norm3 = nn.BatchNorm1d(hidden_size2)
        #self.dropout3 = nn.Dropout(drop_rate2)
        self.dense3 = nn.utils.weight_norm(nn.Linear(hidden_size2, hidden_size))
        
      #  self.batch_norm4 = nn.BatchNorm1d(hidden_size)
      #  self.dropout4 = nn.Dropout(drop_rate3)
        self.dense4 = nn.utils.weight_norm(nn.Linear(hidden_size, num_targets))

        
    def forward(self, x,mode='DAE'):
      #  x = self.batch_norm1(x)
       # x1 = self.dropout1(x1)
        x1 = F.relu(self.dense1(x))
        
            
        x2 = self.batch_norm2(x1)
      #  x = self.dropout2(x)
        x2 = F.relu(self.dense2(x2))
        
        x3 = self.batch_norm3(x2)
      
        x3 = F.relu(self.dense3(x3))
        
        out = self.dense4(x3)
        
        if mode == 'DAE':
            return out
        else:
            return x1,x2,x3



