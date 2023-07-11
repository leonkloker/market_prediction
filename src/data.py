import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import yfinance

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

class MultiStockData(Dataset):
    def __init__(self, stocks, period, use_interval=[0,1]):
        self.stocks = stocks
        self.interval = use_interval
        self.data = [yfinance.Ticker(stock).history(period=period) for stock in stocks]
        self.data = [np.array(data[['Open', 'High', 'Low', 'Close', 'Volume']], dtype=np.float32) for data in self.data]
        self.length = len(self.data)

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        data = self.data[idx]
        x = torch.tensor(data[int(self.interval[0] * data.shape[0]) : min(int(self.interval[1] * data.shape[0]),
                                                                           data.shape[0]-1), :])
        y = torch.tensor(data[int(self.interval[0] * data.shape[0] + 1) : 
                              min(int(self.interval[1] * data.shape[0] + 1), data.shape[0]), 3]).unsqueeze(-1)
        return x, y