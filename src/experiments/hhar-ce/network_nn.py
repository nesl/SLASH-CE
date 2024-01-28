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
            nn.Conv1d(in_channels=6, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=3),
            nn.BatchNorm1d(num_features=64),
            nn.Dropout(drop_out),
  
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=3),
            nn.BatchNorm1d(num_features=64),
            nn.Dropout(drop_out),

            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=3),
            nn.BatchNorm1d(num_features=64),
            nn.Dropout(drop_out)
        )
  
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=192, out_features=84),
            nn.ReLU(),
            nn.Linear(84, self.n_class),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
        
  
    def forward(self, x, marg_idx=None, type=1):
        x = self.conv(x)
        x = self.fc(x)
        return x