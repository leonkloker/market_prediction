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
    def __init__(self, stocks, period, start=0, end=1, binary=False):
        self.stocks = stocks
        self.binary = binary
        self.interval = [start, end]
        self.data_pd = [yfinance.Ticker(stock).history(period=period) for stock in stocks]
        self.data = [np.array(data[['Volume', 'High', 'Low', 'Close']], dtype=np.float32) for data in self.data_pd]
        self.open = [np.array(data[['Open']], dtype=np.float32) for data in self.data_pd]

        # turn high, low and close into percentage change relative to open
        for i in range(len(self.data)):
            self.data[i][:,1:] = (self.data[i][:,1:] - self.open[i]) / self.open[i]

        # normalize volume
        for i in range(len(self.data)):
            self.data[i][:,0] = self.data[i][:,0] - np.mean(self.data[i][:,0])
            self.data[i][:,0] = self.data[i][:,0] / np.std(self.data[i][:,0])

        self.length = len(self.data)

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        data = self.data[idx]
        x = torch.tensor(data[int(self.interval[0] * data.shape[0]) : min(int(self.interval[1] * data.shape[0]),
                                                                           data.shape[0]-1), :])
        y = torch.tensor(data[int(self.interval[0] * data.shape[0] + 1) : 
                              min(int(self.interval[1] * data.shape[0] + 1), data.shape[0]), -1]).unsqueeze(-1)
        if self.binary:
            y = (y > 0).float()
            y[y == 0] = -1

        return x, y