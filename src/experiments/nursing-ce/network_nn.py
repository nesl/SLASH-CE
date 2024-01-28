import torch
import torch.nn as nn
from torch.nn.modules import dropout
import torch.nn.functional as F
from torch.nn.modules.activation import Softmax

class Net_nn(nn.Module):
    def __init__(self, n_class, drop_out=0.5):
        super().__init__()
        self.n_class = n_class
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.BatchNorm1d(num_features=64),
            nn.Dropout(drop_out),
  
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.BatchNorm1d(num_features=64),
            nn.Dropout(drop_out),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.BatchNorm1d(num_features=64),
            nn.Dropout(drop_out)
        )
  
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(32, self.n_class),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
        
  
    def forward(self, x, marg_idx=None, type=1):
        x = self.conv(x)
        x = self.fc(x)
        # print(x.shape)
        return x