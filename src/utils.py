import math
import numpy as np
import torch

from torch import nn

def generate_mask(sz1, sz2=None, window=-1):
    """Generate attention masks for the transformer.

    Args:
        sz1 (int): Amount of tokens in the sequence.
        sz2 (int, optional): Amount of tokens to attend to. (Same as sz1 if None) 
                Defaults to None.
        window (int, optional): Window size of the attention mask. (-1 for full causal attention,
                -2 for full attention) Defaults to -1.

    Returns:
        torch.tensor: attention mask of size (sz1, sz2)
    """
    
    # square mask
    if sz2 is None:
        sz2 = sz1
    
    # no mask
    if window == -2:
        mask = None

    # mask for full causal attention
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
    """Generate attention masks for the transformer.

    Args:
        sz1 (int): Amount of tokens in the sequence.
        sz2 (int, optional): Amount of tokens to attend to. (Same as sz1 if None) 
                Defaults to None.
        window (int, optional): Window size of the attention mask. (-1 for full causal attention,
                -2 for full attention) Defaults to -1.

    Returns:
        torch.tensor: attention mask of size (sz1, sz2)
    """
    
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
    """Positional encoding used in the Attention is All you need paper.

    """
    def __init__(self, d_model, dropout=0.0, max_len=15000):
        """Constructor method.

        Args:
            d_model (int): Embedding size of transformer.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            max_len (int, optional): Maximum sequence length. Defaults to 15000.
        """
        
        super(PositionalEncodingNLP, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.tensor): Input sequence to transformer.

        Returns:
            torch.tensor: Input sequence with positional encoding added.
        """
        x = x + self.pe[:x.size(-2), :]
        return self.dropout(x)
    
class PositionalEncodingLinear(nn.Module):
    """Linear positional encoding.

    """
    def __init__(self, d_model, max_len=15000, dropout=0.0):
        """Constructor method.

        Args:
            d_model (int): Embedding size of transformer.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            max_len (int, optional): Maximum sequence length. Defaults to 15000.
        """
        
        super(PositionalEncodingLinear, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:,:] = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) / max_len
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.tensor): Input sequence to transformer.

        Returns:
            torch.tensor: Input sequence with positional encoding added.
        """
        x = x + self.pos_encoding[:x.size(-2), :]
        return self.dropout(x)
    
class PositionalEncodingSinusoidal(nn.Module):
    """Sinusoidal positional encoding.

    """
    def __init__(self, d_model, max_len=15000, dropout=0.0):
        """Constructor method.

        Args:
            d_model (int): Embedding size of transformer.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            max_len (int, optional): Maximum sequence length. Defaults to 15000.
        """
        
        super(PositionalEncodingSinusoidal, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float)
        pos_encoding[:,:] = -torch.cos(torch.pi * (position / max_len)).unsqueeze(1)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.tensor): Input sequence to transformer.

        Returns:
            torch.tensor: Input sequence with positional encoding added.
        """
        x = x + self.pos_encoding[:x.size(-2), :]
        return self.dropout(x)
    
class PositionalEncodingLearned(nn.Module):
    """Learned positional encoding.

    """
    def __init__(self, d_model, max_len=15000, dropout=0.0):
        """Constructor method.

        Args:
            d_model (int): Embedding size of transformer.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            max_len (int, optional): Maximum sequence length. Defaults to 15000.
        """
        super(PositionalEncodingLearned, self).__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        self.linear_pos_encoding = PositionalEncodingLinear(d_model, max_len, dropout)
        self.linear = nn.Sequential(nn.Linear(d_model, d_model), 
                                    nn.ReLU(), 
                                    nn.Linear(d_model, d_model),
                                    nn.Tanh())

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.tensor): Input sequence to transformer.

        Returns:
            torch.tensor: Input sequence with positional encoding added.
        """
        x = x + self.linear(self.linear_pos_encoding.pos_encoding)[:x.size(-2), :]
        return self.dropout(x)
    
class Time2Vec(nn.Module):
    def __init__(self, k, dropout=0.0, d_time=1):
        """Time2Vec temporal encoding.

        Args:
            k (int): Dimension of the temporal encoding.
            dropout (float, optional): Dropout rate. Defaults to 0.0.
            d_time (int, optional): Amount of input features to temporal encoding.
                    Defaults to 1.
        """
        super(Time2Vec, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(d_time, k)
        self.d_time = d_time

    def forward(self, x, t=None):
        """Forward pass.

        Args:
            x (torch.tensor): Input sequence to transformer.
            t (torch.tensor, optional): Time features. Defaults to None.

        Returns:
            torch.tensor: Input sequence with positional encoding appended
                        in embedding dimension.
        """
        
        # if the time features are not provided, use the index of the sequence or a list
        if not torch.is_tensor(t):
            if t == None:
                t = torch.arange(x.size(-2), dtype=torch.float32).unsqueeze(-1)
                t = t.repeat(1, self.d_time)
            else:
                t = torch.tensor(t).unsqueeze(-1)

            t = t.to(x.device)
            t = self.linear(t)
            t = torch.cat([t[:, 0].unsqueeze(-1), torch.sin(t[:, 1:])], dim=-1)
            t = t.unsqueeze(0).repeat(x.size(0), 1, 1)
            x = torch.cat([x, t], dim=-1)

        # if the time features are provided, use them
        else:
            t = t.to(x.device)
            t = self.linear(t)
            t = torch.cat([t[:, :, 0].unsqueeze(-1), torch.sin(t[:, :, 1:])], dim=-1)
            x = torch.cat([x, t], dim=-1)

        return x

def accuracy(out, y, torch=True, data='normalized', binary=False):
    """FUnction to calculate the binary accuracy of the model.

    Args:
        out (torch.tensor): Model output sequence.
        y (torch.tensor): Label sequence.
        torch (bool, optional): If true, out and y are expected to be torch tensors instead of numpy ndarrays.
                    Defaults to True.
        data (str, optional): 'percent' or 'normalized. Defaults to 'normalized'.
        binary (bool, optional): If transformer predicts binary stock movement or actual price.
                    Defaults to False.

    Returns:
        float: Binary accuracy of the model.
    """
    
    if torch:
        out = out.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        
    out = np.squeeze(out)
    y = np.squeeze(y)

    if not binary:
        if data == 'normalized':
            out = out[1:] - y[:-1]
            y = y[1:] - y[:-1]

        out = (out > 0).astype(int)
        y = (y > 0).astype(int)
    
    if binary:
        out = (out > 0.5).astype(int)
        y = (y > 0.5).astype(int)
        
    return np.mean(out == y)

def trading_strategy(out, y, binary=False, data='percent'):
    """Function to calculate the trading signals based on the model outputs.

    Args:
        out (torch.tensor): Output sequence of the transformer.
        y (torch.tensor): Label sequence.
        binary (bool, optional): If transformer predicts binary stock movement or actual price.
                    Defaults to False.
        data (str, optional): 'percent' or 'normalized. Defaults to 'normalized'.

    Returns:
        torch.tensor: Contains -1 or 1 for short or long positions for each day.
    """
    out = np.squeeze(out)
    if binary:
        signal = (out > 0.5).astype(float)
        signal[signal == 0] = -1.
    elif data == 'percent':
        signal = np.zeros_like(out)
        signal[out > 0] = 1
        signal[out < 0] = -1
    elif data == 'normalized':
        signal = np.zeros_like(out)
        signal[1:] = out[1:] - y[:-1]
        signal[signal > 0] = 1
        signal[signal < 0] = -1
    return signal

def get_net_value(signals, y, data='percent', mean=-1, std=-1):
    """Function to calculate the net value of the trading strategy.

    Args:
        signals (torch.tensor): Trading signals from the trading strategy.
        y (torch.tensor): Label sequence.
        data (str, optional): 'percent' or 'normalized. Defaults to 'percent'.
        mean (int, optional): Mean used to normalize the closing prices, needs to be specified when
                data is 'normalized'. Defaults to -1.
        std (int, optional): Std used to normalize the closing prices, needs to be specified
                when data is 'normalized'. Defaults to -1.

    Raises:
        ValueError: When data is 'normalized' but mean and std are not provided.

    Returns:
        numpy.ndarray: Net value of the trading strategy at each trading day.
    """
    
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
    """Function to calculate the maximum drawdown of the trading strategy.

    Args:
        net_value (numpy.ndarray): Net value of the trading strategy at each trading day.

    Returns:
        float: Maximum drawdown of the trading strategy.
    """
    
    net_value = np.squeeze(net_value)
    i = np.argmax(np.maximum.accumulate(net_value) - net_value)
    if i > 0:
        j = np.argmax(net_value[:i])
    else:
        j = 0 
    return (net_value[j] - net_value[i]) / (net_value[j] + 1)
