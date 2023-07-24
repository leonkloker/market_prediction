import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import yfinance

# Include features such as running average 5, 10, 20, 50 days of all features
# Running volatility 5, 10, 20, 50 days of all features

class MultiStockData(Dataset):
    def __init__(self, stocks, period, start=0, end=1, binary=False):
        self.stocks = stocks
        self.binary = binary
        self.interval = [start, end]
        self.data_pd = [yfinance.Ticker(stock).history(period=period) for stock in stocks]
        self.data = [np.array(data[['Volume', 'High', 'Low', 'Close']], dtype=np.float32) for data in self.data_pd]
        self.open = [np.array(data[['Open']], dtype=np.float32) for data in self.data_pd]
        
        for i in range(len(self.data)):
            additional_features = []

            # normalize volume
            self.data[i][:,0] = (self.data[i][:,0] - np.mean(self.data[i][:,0])) / np.std(self.data[i][:,0])

            # turn high, low and close into percentage change relative to open
            self.data[i][:,1:] = (self.data[i][:,1:] - self.open[i]) / self.open[i]

            #calculate running mean and std of all features and add them as additional features
            for n in [5, 10, 20, 50]:
                for j in range(self.data[i].shape[1]):

                    # running average
                    additional_features.append(running_average(self.data[i][:,j], n))
                    run_std = running_average(self.data[i][:,j]**2, n)
                    run_std = np.sqrt(np.maximum(run_std - additional_features[-1]**2, 0))

                    # running volatility
                    additional_features.append(run_std)
            
            additional_features = np.array(additional_features).T
            self.data[i] = np.concatenate([additional_features, self.data[i]], axis=1)

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

class MultiStockDataNormalized(Dataset):
    def __init__(self, stocks, period, start=0, end=1, binary=False):
        self.stocks = stocks
        self.binary = binary
        self.interval = [start, end]
        self.data_pd = [yfinance.Ticker(stock).history(period=period) for stock in stocks]
        self.data = [np.array(data[['Volume', 'High', 'Low', 'Open', 'Close']], dtype=np.float32) for data in self.data_pd]

        # normalize data
        for i in range(len(self.data)):
            additional_features = []

            #calculate running mean and std of all features and add them as additional features
            for n in [5, 10, 20, 50]:
                for j in range(self.data[i].shape[1]):
                    self.data[i] = (self.data[i] - np.mean(self.data[i], axis=0)) / np.std(self.data[i], axis=0)
                    additional_features.append(running_average(self.data[i][:,j], n))
                    run_std = running_average(self.data[i][:,j]**2, n)
                    run_std = np.sqrt(np.maximum(run_std - additional_features[-1]**2, 0))
                    additional_features.append(run_std)

            additional_features = np.array(additional_features).T
            self.data[i] = np.concatenate([self.data[i], additional_features], axis=1)

        self.length = len(self.data)

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        data = self.data[idx]
        x = torch.tensor(data[int(self.interval[0] * data.shape[0]) : min(int(self.interval[1] * data.shape[0]), data.shape[0]-1), :])
        
        y = torch.tensor(data[int(self.interval[0] * data.shape[0] + 1) : 
                              min(int(self.interval[1] * data.shape[0] + 1), data.shape[0]), -1]).unsqueeze(-1)
        if self.binary:
            y = data[int(self.interval[0] * data.shape[0]) : min(int(self.interval[1] * data.shape[0]) + 1, data.shape[0]), :]
            y = np.convolve(y, np.array([1, -1]), mode='valid')
            y = (y > 0).float()

        return x, y
    
def running_average(x, n, mode='same'):
    return np.array(np.convolve(x, np.ones((n,))/n, mode=mode), dtype=np.float32)
