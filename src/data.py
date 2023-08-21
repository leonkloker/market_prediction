import datetime
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
import torch
import yfinance

# Normalize every stock price with the runnaing average and standard deviation of the last 10 days e.g. to
# account for the distribution shift over time
class StockData(Dataset):
    def __init__(self, stocks, period, start=0, end=1, data='normalized', 
                 binary=False, features=['Close'], additional_features=[],
                 normalization_mask=[], normalization_window=-1, time=False):
        self.stocks = stocks
        self.binary = binary
        self.interval = [start, end]
        self.mode = data
        self.period = period

        if normalization_mask == []:
            normalization_mask = [True] * len(features)

        features.append(features.pop(features.index('Close')))
        self.features = features
        self.additional_features = additional_features
        self.data_pd = [yfinance.Ticker(stock).history(period=period) for stock in stocks]
        
        self.length = len(stocks)
        self.samples_per_stock = 1
        max_seq_len = max([len(stock) for stock in self.data_pd])
        if max_seq_len > 20000:
            self.samples_per_stock = max_seq_len // 2000 + 1
            self.length = self.length * self.samples_per_stock

        self.data = [np.array(data[self.features], dtype=np.float32) for data in self.data_pd]
        self.weekdays = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
        self.time = time
        self.times = []
        self.mean = []
        self.std = []

        if self.time:
            for i in range(len(self.data)):
                times = []
                for j in range(self.data_pd[i].shape[0]):
                    string = str(self.data_pd[i].index[j])[:-6]
                    date = datetime.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')
                    weekday = self.weekdays[date.strftime('%a')]
                    date = [(date.year - 2010) / 10, date.month / 12, date.day / 31, weekday / 4]
                    times.append(date)
                self.times.append(np.array(times, dtype=np.float32))

        for i in range(len(self.stocks)):
            additional_features = []

            if self.mode == 'normalized':
                if normalization_window == -1:
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

                if len(additional_features) > 0:
                    additional_features = np.array(additional_features).T
                    self.data[i] = np.concatenate([additional_features, self.data[i]], axis=1)

            if self.mode == 'percent':
                for j in range(self.data[i].shape[1]):
                    if normalization_mask[j]:
                        self.data[i][1:,j] = np.convolve(self.data[i][:,j], np.array([1, -1]), mode='valid') / self.data[i][:-1,-1]
                        self.data[i][0,j] = 0
                    else:
                        self.data[i][:,j] = (self.data[i][:,j] - np.mean(self.data[i][:,j])) / np.std(self.data[i][:,j])

                for n in self.additional_features:
                    for j in range(self.data[i].shape[1]):
                        additional_features.append(running_average(self.data[i][:,j], n))
                        run_std = running_average(self.data[i][:,j]**2, n)
                        run_std = np.sqrt(np.maximum(run_std - additional_features[-1]**2, 0))
                        additional_features.append(run_std)
                
                if len(additional_features) > 0:
                    additional_features = np.array(additional_features).T
                    self.data[i] = np.concatenate([additional_features, self.data[i]], axis=1)

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        split_idx = idx % self.samples_per_stock
        idx = idx // self.samples_per_stock
        sample = self.data[idx]
        x = torch.tensor(sample[int(self.interval[0] * sample.shape[0]) : min(int(self.interval[1] * sample.shape[0]), sample.shape[0]-1), :])

        y = torch.tensor(sample[int(self.interval[0] * sample.shape[0] + 1) : min(int(self.interval[1] * sample.shape[0] + 1), sample.shape[0]), -1]).unsqueeze(-1)
        
        if self.binary:
            if self.mode == 'percent':
                y = (y > 0).float()
            
            if self.mode == 'normalized':
                y = sample[int(self.interval[0] * sample.shape[0]) : min(int(self.interval[1] * sample.shape[0] + 1), sample.shape[0]), -1]
                y = np.convolve(y, np.array([1, -1]), mode='valid')
                y = torch.tensor(y).unsqueeze(-1)
                y = (y > 0).float()

        if self.time:
            t = torch.tensor(self.times[idx][int(self.interval[0] * sample.shape[0]) : min(int(self.interval[1] * sample.shape[0]), sample.shape[0]-1), :])
        else:
            t = None
            
        if self.samples_per_stock > 1:
            start_idx = int((split_idx / self.samples_per_stock) * x.shape[0])
            end_idx = int(((split_idx + 1) / self.samples_per_stock) * x.shape[0])
            x = x[start_idx : end_idx, :]
            y = y[start_idx : end_idx, :]
            t = t[start_idx : end_idx, :]
 
        return x, y, t

def running_average(x, n):
    res = np.zeros_like(x, dtype=np.float32)
    res[:n-1] = np.cumsum(x[:n-1]) / np.arange(1, n)
    res[n-1:] = np.convolve(x, np.ones((n,))/n, mode='valid')
    return res
