import numpy as np
import torch
from torch import nn
import math

def generate_mask(sz1, sz2=None, window=-1):
        # square mask
        if sz2 is None:
            sz2 = sz1
        
        # no mask
        if window == -2:
            mask = None

        # mask when all past history is available
        elif window == -1:
            mask = (torch.tril(torch.ones(sz1, sz2)) == 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        # mask when only a window of past history is available
        else:
            mask = torch.zeros(sz1, sz2)
            for i in range(sz1):
                mask[i, max(0, i - window + 1) : min(i + 1, sz2)] = 1
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask

def generate_mask_bool(sz1, sz2=None, window=-1):
        # square mask
        if sz2 is None:
            sz2 = sz1
        
        # no mask
        if window == -2:
            mask = None

        # mask when all past history is available
        elif window == -1:
            mask = torch.logical_not((torch.tril(torch.ones(sz1, sz2)) == 1).bool())
        
        # mask when only a window of past history is available
        else:
            mask = torch.zeros(sz1, sz2)
            for i in range(sz1):
                mask[i, max(0, i - window + 1) : min(i + 1, sz2)] = 1
            mask = torch.logical_not(mask.bool())

        return mask

class PositionalEncodingNLP(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=15000):
        super(PositionalEncodingNLP, self).__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(-2), :]
        return self.dropout(x)
    
class PositionalEncodingLinear(nn.Module):
    def __init__(self, d_model, max_len=15000, dropout=0.0):
        super(PositionalEncodingLinear, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:,:] = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) / max_len
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        x = x + self.pos_encoding[:x.size(-2), :]
        return self.dropout(x)
    
class PositionalEncodingSinusoidal(nn.Module):
    def __init__(self, d_model, max_len=15000, dropout=0.0):
        super(PositionalEncodingSinusoidal, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float)
        pos_encoding[:,:] = -torch.cos(torch.pi * (position / max_len)).unsqueeze(1)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        x = x + self.pos_encoding[:x.size(-2), :]
        return self.dropout(x)
    
class PositionalEncodingLearned(nn.Module):
    def __init__(self, d_model, max_len=15000, dropout=0.0):
        super(PositionalEncodingLearned, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.linear_pos_encoding = PositionalEncodingLinear(d_model, max_len, dropout)
        self.linear = nn.Sequential(nn.Linear(d_model, d_model), 
                                    nn.ReLU(), 
                                    nn.Linear(d_model, d_model),
                                    nn.Tanh())

    def forward(self, x):
        x = x + self.linear(self.linear_pos_encoding.pos_encoding)[:x.size(-2), :]
        return self.dropout(x)
    
class Time2Vec(nn.Module):
    def __init__(self, k, dropout=0.0):
        super(Time2Vec, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(1, k)

    def forward(self, x, t=None):
        if t == None:
            t = torch.arange(x.size(-2), dtype=torch.float32).unsqueeze(-1)
        else:
            t = torch.tensor(t).unsqueeze(-1)

        t = t.to(x.device)
        t = self.linear(t)
        t = torch.cat([t[:, 0].unsqueeze(-1), torch.sin(t[:, 1:])], dim=-1)
        t = t.unsqueeze(0).repeat(x.size(0), 1, 1)
        x = torch.cat([x, t], dim=-1)
        return x

def accuracy(out, y, torch=True, data='normalized'):
    if torch:
        out = out.cpu().detach().numpy()
        y = y.cpu().detach().numpy()

    if data == 'normalized':
        out = np.convolve(np.squeeze(out), np.array([1, -1]), mode='valid')
        y = np.convolve(np.squeeze(y), np.array([1, -1]), mode='valid')
    
    out = (out > 0).astype(int)
    y = (y > 0).astype(int)
    return np.mean(out == y)

def trading_strategy(predictions, binary=False, data='percent'):
    predictions = np.squeeze(predictions)
    if binary:
        signal = (predictions > 0.5).astype(int)
        signal[signal == 0] = -1

    if data == 'percent':
        signal = np.zeros_like(predictions)
        signal[predictions > 0] = 1
        signal[predictions < 0] = -1

    if data == 'normalized':
        signal = np.convolve(predictions, np.array([1, -1]), mode='same')
        signal[signal > 0] = 1
        signal[signal < 0] = -1
    
    return signal

def get_net_value(signals, y, data='percent', mean=-1, std=-1):
    signals = np.squeeze(signals)
    y = np.squeeze(y)

    if data == 'percent':
        net_value = np.cumsum(signals * y)
    
    if data == 'normalized':
        if mean == -1 or std == -1:
            raise ValueError('mean and std must be provided for normalized data')
        
        enum = np.zeros_like(y)
        enum[1:] = np.convolve(y, np.array([1, -1]), mode='valid')
        enum[1:] = enum[1:] / (y[:-1] + (mean/std))
        net_value = np.cumsum(signals * enum)
    
    return net_value

def get_max_drawdown(net_value):
    net_value = np.squeeze(net_value)
    i = np.argmax(np.maximum.accumulate(net_value) - net_value)
    if i > 0:
        j = np.argmax(net_value[:i])
    else:
        j = 0 
    return (net_value[j] - net_value[i]) / (net_value[j] + 1)