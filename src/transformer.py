import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, d_features, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.0, window=20):
        super(Transformer, self).__init__()

        self.embedding = nn.Linear(d_features, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.regressor = nn.Sequential(nn.Linear(d_model, 256),
                                      nn.Dropout(dropout),
                                      nn.ReLU(),
                                      nn.Linear(256, 128),
                                      nn.Dropout(dropout),
                                      nn.ReLU(),
                                      nn.Linear(128, 1))
        self.tgt_mask = None
        self.src_mask = None
        self.memory_mask = None
        self.window = window

    def generate_mask(self, sz, window=-1):
        # mask when all past history is available
        if window == -1:
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        # mask when only a window of past history is available
        else:
            mask = torch.zeros(sz, sz)
            for i in range(sz):
                mask[i, max(0, i - window + 1) : min(i + 1, sz)] = 1
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            
        return mask
        
    def forward(self, src, tgt):
        if self.tgt_mask is None or self.tgt_mask.size(0) != tgt.size(0):
            self.tgt_mask = self.generate_mask(tgt.size(0), window=self.window).to(tgt.device)
        
        if self.src_mask is None or self.src_mask.size(0) != src.size(0):
            self.src_mask = self.generate_mask(src.size(0)).to(src.device)

        if self.memory_mask is None or self.memory_mask.size(0) != src.size(0):
            self.memory_mask = self.generate_mask(src.size(0)).to(src.device)

        src = self.embedding(src)
        tgt = self.embedding(tgt)
        features = self.transformer(src, tgt, src_mask=self.src_mask, tgt_mask=self.tgt_mask, memory_mask=self.memory_mask)
        output = self.regressor(features)
        return output
