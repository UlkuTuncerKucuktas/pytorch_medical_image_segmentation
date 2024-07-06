import torch
import torch.nn as nn

import torch
import torch.nn as nn

class SegmentationModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(SegmentationModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        enc_out, skip_connections = self.encoder(x)
        dec_out = self.decoder(enc_out, skip_connections)
        return dec_out
