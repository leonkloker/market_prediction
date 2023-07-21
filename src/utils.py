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
        if window == -1:
            mask = (torch.tril(torch.ones(sz1, sz2)) == 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        # mask when only a window of past history is available
        else:
            mask = torch.zeros(sz1, sz2)
            for i in range(sz1):
                mask[i, max(0, i - window + 1) : min(i + 1, sz2)] = 1
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

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
    
def accuracy(out, y):
    out = out.cpu().detach().numpy()
    y = y.cpu().detach().numpy()
    pred = np.apply_along_axis(lambda m: np.convolve(m, np.array([1, -1]), mode='valid'), axis=1, arr=out)
    pred = (pred > 0).astype(int)
    ground_truth = np.apply_along_axis(lambda m: np.convolve(m, np.array([1, -1]), mode='valid'), axis=1, arr=y)
    ground_truth = (ground_truth > 0).astype(int)
    return np.mean(pred == ground_truth)
