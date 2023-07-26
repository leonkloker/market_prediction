import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import yfinance

class StockData(Dataset):
    def __init__(self, stocks, period, start=0, end=1, data='normalized', 
                 binary=False, features=['Close'], additional_features=[],
                 normalization_mask=[]):
        self.stocks = stocks
        self.binary = binary
        self.interval = [start, end]
        self.mode = data
        self.length = len(stocks)

        if period == 'max':
            period = '8000d'
        self.period = period

        if normalization_mask == []:
            normalization_mask = [True] * len(features)

        features.append(features.pop(features.index('Close')))
        self.features = features
        self.additional_features = additional_features
        self.data = [yfinance.Ticker(stock).history(period=period) for stock in stocks]
        self.data = [np.array(data[self.features], dtype=np.float32) for data in self.data]
        self.mean = []
        self.std = []

        for i in range(len(self.stocks)):
            additional_features = []

            if self.mode == 'normalized':
                self.mean.append(np.mean(self.data[i][:, -1]))
                self.std.append(np.std(self.data[i][:, -1]))
                
                for j in range(self.data[i].shape[1]):
                    if normalization_mask[j]:
                        self.data[i][:,j] = (self.data[i][:,j] - self.mean[-1]) / self.std[-1]
                    else:
                        self.data[i][:,j] = (self.data[i][:,j] - np.mean(self.data[i][:,j])) / np.std(self.data[i][:,j])

                for n in self.additional_features:
                    for j in range(self.data[i].shape[1]):
                        additional_features.append(running_average(self.data[i][:,j], n))
                        run_std = running_average(self.data[i][:,j]**2, n)
                        run_std = np.sqrt(np.maximum(run_std - additional_features[-1]**2, 0))
                        additional_features.append(run_std)
            
                additional_features = np.array(additional_features).T
                self.data[i] = np.concatenate([additional_features, self.data[i]], axis=1)

            if self.mode == 'percent':
                for j in range(self.data[i].shape[1]):
                    if normalization_mask[j]:
                        self.data[i][:,j] = np.convolve(self.data[i][:,j], np.array([1, -1]), mode='valid') / self.data[i][:-1,-1]
                    else:
                        self.data[i][:,j] = (self.data[i][:,j] - np.mean(self.data[i][:,j])) / np.std(self.data[i][:,j])

                for n in self.additional_features:
                    for j in range(self.data[i].shape[1]):
                        additional_features.append(running_average(self.data[i][:,j], n))
                        run_std = running_average(self.data[i][:,j]**2, n)
                        run_std = np.sqrt(np.maximum(run_std - additional_features[-1]**2, 0))
                        additional_features.append(run_std)
                
                additional_features = np.array(additional_features).T
                self.data[i] = np.concatenate([additional_features, self.data[i]], axis=1)

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        x = torch.tensor(sample[int(self.interval[0] * sample.shape[0]) : min(int(self.interval[1] * sample.shape[0]), sample.shape[0]-1), :])

        y = torch.tensor(sample[int(self.interval[0] * sample.shape[0] + 1) : min(int(self.interval[1] * sample.shape[0] + 1), sample.shape[0]), -1]).unsqueeze(-1)
        
        if self.binary:
            if self.mode == 'percent':
                y = (y > 0).float()
            
            if self.mode == 'normalized':
                y = sample[int(self.interval[0] * sample.shape[0]) : min(int(self.interval[1] * sample.shape[0]) + 1, sample.shape[0]), -1].unsqueeze(-1)
                y = np.convolve(y, np.array([1, -1]), mode='valid')
                y = (y > 0).float()

        return x, y

def running_average(x, n, mode='same'):
    return np.array(np.convolve(x, np.ones((n,))/n, mode=mode), dtype=np.float32)
