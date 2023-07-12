import torch
from torch import nn

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

class PositionalEncodingStandard(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super(PositionalEncodingStandard, self).__init__()

        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        # x has shape (batch_size, seq_len, n_inp)
        return x + self.pos_encoding[:x.size(-2), :]
    