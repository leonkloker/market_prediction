import torch
import torch.nn as nn

from utils import PositionalEncodingStandard, generate_mask

class Transformer(nn.Module):
    def __init__(self, d_features, d_model, n_head, n_encoder_layers, n_decoder_layers, 
                 d_feedforward, dropout=0.0, activation='gelu'):
        super(Transformer, self).__init__()

        self.tgt_mask = generate_mask(1)
        self.src_mask = generate_mask(1)
        self.memory_mask = generate_mask(1)
        self.enc_window = -1
        self.dec_window = -1
        self.mem_window = -1
        self.decoder_only = False

        self.embedding = nn.Sequential(nn.Linear(d_features, d_model),
                                       PositionalEncodingStandard(d_model))
        if n_encoder_layers == 0:
            self.transformer = nn.Transformer(d_model, n_head, n_encoder_layers, n_decoder_layers,
                                           d_feedforward, dropout, custom_encoder=MyIdentity(),
                                           batch_first=True, activation=activation)
        else:
            self.transformer = nn.Transformer(d_model, n_head, n_encoder_layers, n_decoder_layers,
                                           d_feedforward, dropout, batch_first=True, activation=activation)
            
        self.regressor = nn.Sequential(nn.Linear(d_model, int(d_model/2)),
                                      nn.Dropout(dropout),
                                      nn.ReLU(),
                                      nn.Linear(int(d_model/2), 1))
        
    def forward(self, src, tgt, enc_window=-1, dec_window=-1, mem_window=-1):
        # create source mask if sequence length or window has changed
        if self.src_mask.size(0) != src.size(-2) or self.enc_window != enc_window:
            mask = generate_mask(src.size(-2), window=enc_window).to(src.device)
            self.src_mask = mask
            self.enc_window = enc_window

        # create target mask if sequence length or window has changed
        if self.tgt_mask.size(0) != tgt.size(-2) or self.dec_window != dec_window:
            mask = generate_mask(tgt.size(-2), window=dec_window).to(tgt.device)
            self.tgt_mask = mask
            self.dec_window = dec_window
        
        # create memory mask if sequence length or window has changed
        if self.memory_mask.size(0) != tgt.size(-2) or self.mem_window != mem_window:
            mask = generate_mask(tgt.size(-2), src.size(-2), window=mem_window).to(tgt.device)
            self.memory_mask = mask
            self.mem_window = mem_window

        src = self.embedding(src)
        tgt = self.embedding(tgt)

        output = self.transformer(src, tgt, tgt_mask=self.tgt_mask, src_mask=self.src_mask, memory_mask=self.memory_mask)
        output = self.regressor(output)

        return output

class MyIdentity(nn.Module):
    def __init__(self):
        super(MyIdentity, self).__init__()

    def forward(self, x, **kwargs):
        return x
