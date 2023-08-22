import datetime
import numpy as np
import os
import pandas as pd
import torch
import yfinance

from torch.utils.data import Dataset

class StockData(Dataset):
    """Class for loading stock data from yfinance.

    """
    
    def __init__(self, stocks, period, start=0, end=1, data='normalized', 
                 binary=False, features=['Close'], additional_features=[],
                 normalization_mask=[], time=False):
        """Constructor of the StockData class.
        
        Args:
            stocks (list): Contains the ticker symbols of the stocks to be loaded as strings.
            period (string): Describes how far back in time the data should be loaded.
            start (int, optional): Fraction where the data should start relative
                                to all loaded data. Defaults to 0.
            end (int, optional): Fraction where the data should end relative
                                to all loaded data. Defaults to 1.
            data (str, optional): 'normalized' or 'percent', describes the normalization
                                scheme that is used. 'percent' means all price data is 
                                calculated as change to the previous closing price.
                                'normalized' means all price data is normalized with the
                                mean and standard deviation of all closing prices. 
                                Defaults to 'normalized'.
            binary (bool, optional): Whether the task should be to predict the next 
                                closing price (False) or only if the price increases
                                or decreases (True). Defaults to False.
            features (list, optional): List of all the features that should be loaded
                                from yfinance. Defaults to ['Close'].
            additional_features (list, optional): List of amount of days for which the 
                                moving averages and volatilities of all original features
                                should be calculated and also used as additional features.      
                                Defaults to [].
            normalization_mask (list, optional): List of bools describing which of the original
                                features should be normalized with mean and std of the 
                                closing prices (True) or with their own mean and std (False).
                                Defaults to [all True].
            time (bool, optional): If year, month, date and weekday should be used for the 
                                positional encoding of the transformer. Defaults to False.
        """
        
        # initialize class attributes
        self.stocks = stocks
        self.binary = binary
        self.interval = [start, end]
        self.mode = data
        self.period = period

        # set normalization mask if not given
        if normalization_mask == []:
            normalization_mask = [True] * len(features)

        # move closing price to last list position
        features.append(features.pop(features.index('Close')))
        self.features = features
        self.additional_features = additional_features
        
        # load dataframes from yfinance
        self.data_pd = [yfinance.Ticker(stock).history(period=period) for stock in stocks]
        
        # split the timeseries if it is too long
        self.length = len(stocks)
        self.samples_per_stock = 1
        max_seq_len = max([len(stock) for stock in self.data_pd])
        if max_seq_len > 20000:
            self.samples_per_stock = max_seq_len // 2000 + 1
            self.length = self.length * self.samples_per_stock

        # convert dataframes to numpy arrays
        self.data = [np.array(data[self.features], dtype=np.float32) for data in self.data_pd]
        
        self.weekdays = {'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3, 'Fri': 4, 'Sat': 5, 'Sun': 6}
        self.time = time
        self.times = []
        self.mean = []
        self.std = []

        # calculate temporal features for positional encoding
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

        # calculate features
        for i in range(len(self.stocks)):
            additional_features = []

            if self.mode == 'normalized':
                self.mean.append(np.mean(self.data[i][:, -1]))
                self.std.append(np.std(self.data[i][:, -1]))
                
                # normalize data
                for j in range(self.data[i].shape[1]):
                    if normalization_mask[j]:
                        self.data[i][:,j] = (self.data[i][:,j] - self.mean[-1]) / self.std[-1]
                    else:
                        self.data[i][:,j] = (self.data[i][:,j] - np.mean(self.data[i][:,j])) / np.std(self.data[i][:,j])
                
                # calculate additional features
                for n in self.additional_features:
                    for j in range(self.data[i].shape[1]):
                        additional_features.append(running_average(self.data[i][:,j], n))
                        run_std = running_average(self.data[i][:,j]**2, n)
                        run_std = np.sqrt(np.maximum(run_std - additional_features[-1]**2, 0))
                        additional_features.append(run_std)

                # add additional features to data
                if len(additional_features) > 0:
                    additional_features = np.array(additional_features).T
                    self.data[i] = np.concatenate([additional_features, self.data[i]], axis=1)

            if self.mode == 'percent':
                
                # normalize data
                for j in range(self.data[i].shape[1]):
                    if normalization_mask[j]:
                        self.data[i][1:,j] = np.convolve(self.data[i][:,j], np.array([1, -1]), mode='valid') / self.data[i][:-1,-1]
                        self.data[i][0,j] = 0
                    else:
                        self.data[i][:,j] = (self.data[i][:,j] - np.mean(self.data[i][:,j])) / np.std(self.data[i][:,j])

                # calculate additional features
                for n in self.additional_features:
                    for j in range(self.data[i].shape[1]):
                        additional_features.append(running_average(self.data[i][:,j], n))
                        run_std = running_average(self.data[i][:,j]**2, n)
                        run_std = np.sqrt(np.maximum(run_std - additional_features[-1]**2, 0))
                        additional_features.append(run_std)
                
                # add additional features to data
                if len(additional_features) > 0:
                    additional_features = np.array(additional_features).T
                    self.data[i] = np.concatenate([additional_features, self.data[i]], axis=1)

    def __len__(self):
        """Returns the length of the dataset, i.e. the number of stocks.

        Returns:
            int: Amount of timeseries samples.
        """
        return self.length
        
    def __getitem__(self, idx):
        """Returns the idx-th sample of the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            x (torch.tensor): features of size (sequence length, amount of features)
            y (torch.tensor): labels of size (sequence length, 1)
            t (torch.tensor): time features of size (sequence length, 4)
        """
        
        # calculate sample index
        split_idx = idx % self.samples_per_stock
        idx = idx // self.samples_per_stock
        sample = self.data[idx]
        
        # split sample into features and labels, where labels are shifted to future in time
        x = torch.tensor(sample[int(self.interval[0] * sample.shape[0]) : min(int(self.interval[1] * sample.shape[0]), sample.shape[0]-1), :])
        y = torch.tensor(sample[int(self.interval[0] * sample.shape[0] + 1) : min(int(self.interval[1] * sample.shape[0] + 1), sample.shape[0]), -1]).unsqueeze(-1)
        
        # convert labels to binary if necessary
        if self.binary:
            if self.mode == 'percent':
                y = (y > 0).float()
            
            if self.mode == 'normalized':
                y = sample[int(self.interval[0] * sample.shape[0]) : min(int(self.interval[1] * sample.shape[0] + 1), sample.shape[0]), -1]
                y = np.convolve(y, np.array([1, -1]), mode='valid')
                y = torch.tensor(y).unsqueeze(-1)
                y = (y > 0).float()

        # add time features if necessary
        if self.time:
            t = torch.tensor(self.times[idx][int(self.interval[0] * sample.shape[0]) : min(int(self.interval[1] * sample.shape[0]), sample.shape[0]-1), :])
        else:
            t = None
        
        # split sample into two samples if sequence is too long
        if self.samples_per_stock > 1:
            start_idx = int((split_idx / self.samples_per_stock) * x.shape[0])
            end_idx = int(((split_idx + 1) / self.samples_per_stock) * x.shape[0])
            x = x[start_idx : end_idx, :]
            y = y[start_idx : end_idx, :]
            t = t[start_idx : end_idx, :]
 
        return x, y, t

def running_average(x, n):
    """Calculates the n-day running average of x along the first axis.

    Args:
        x (numpy.ndarray): Timeseries data.
        n (int): Amount of days for running average.

    Returns:
        numpy.ndarray: n-day running average.
    """
    
    res = np.zeros_like(x, dtype=np.float32)
    res[:n-1] = np.cumsum(x[:n-1]) / np.arange(1, n)
    res[n-1:] = np.convolve(x, np.ones((n,))/n, mode='valid')
    return res
