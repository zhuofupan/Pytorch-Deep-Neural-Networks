# -*- coding: utf-8 -*-
import torch
import sys
import numpy as np

from ..dae import Deep_AE

class TRDAE(Deep_AE):
    def __init__(self, **kwargs):
        kwargs['_name'] = 'TRDAE'
        Deep_AE.__init__(self, **kwargs)
    
    def forward(self, x):
        b, m = x.size(0), x.size(1)
        if hasattr(self, 'E_mat') == False: self.E_mat = torch.eye(m).to(self.dvc)
        x_epd = x.view(b, 1, m) * (1 - self.E_mat)
        x_in = x_epd.view(-1, m)
        x_recon = self.decoder(self.encoder(x_in)).view(b,m,m)
        recon = torch.diagonal(x_recon, dim1 = 1, dim2=2)
        if self.task == 'impu':
            self.loss = self._get_impu_loss(recon, x)
        else:
            self.loss = self.L(recon, x)
        return recon
    
    # def forward(self, x):
    #     recon = []
    #     if hasattr(self, 'E_mat') == False: self.E_mat = torch.eye(x.size(1)).to(self.dvc)
    #     for i in range(x.size(0)):
    #         x_in = x[i].view(1,-1) * (1 - self.E_mat)
    #         x_recon = self.decoder(self.encoder(x_in)).diag().view(1,-1)
    #         recon.append(x_recon)
    #     recon = torch.cat(recon)
    #     if self.task == 'impu':
    #         self.loss = self._get_impu_loss(recon, x)
    #     else:
    #         self.loss = self.L(recon, x)
    #     return recon