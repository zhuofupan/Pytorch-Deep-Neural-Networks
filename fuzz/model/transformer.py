# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np

from ..core.module import Module
from ..core.impu_module import Impu_Module
from ..core.layer import MultiHeadAttention, PositionwiseFeedForward

# https://github.com/jadore801120/attention-is-all-you-need-pytorch/blob/master/transformer/Models.py
class EncoderLayer(nn.Module):
    def __init__(self, d_model, v_out, **kwargs):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(d_model, v_out, **kwargs)
        self.pos_ffn = PositionwiseFeedForward(self.slf_attn.v_out, **kwargs)

    def forward(self, q):
        z = self.slf_attn(q, q, q)
        o = self.pos_ffn(z)
        return o

class DecoderLayer(nn.Module):
    def __init__(self, d_model, v_out, enc_attn_k_in, **kwargs):
        super(DecoderLayer, self).__init__()
        kwargs['k_in'] = 0
        self.slf_attn = MultiHeadAttention(d_model, d_model, **kwargs)
        kwargs['k_in'] = enc_attn_k_in
        self.enc_attn = MultiHeadAttention(self.slf_attn.v_out, v_out, **kwargs)
        self.pos_ffn = PositionwiseFeedForward(self.enc_attn.v_out, **kwargs)
        
    # q: former dec_output, k: last enc_output
    def forward(self, q, k):
        z = self.slf_attn(q, q, q)
        z = self.enc_attn(z, k, k)
        o = self.pos_ffn(z)
        return o

class Encoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        
        struct = self.struct
        self.Layer_stack = nn.ModuleList([
            EncoderLayer(struct[i], v_out = struct[i+1], **kwargs)
            for i in range(len(struct)-1)])

    def forward(self, q):
        for Enc_layer in self.Layer_stack:
            q = Enc_layer(q)
        return q

class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])
        struct = self.struct
        struct.reverse()
        enc_attn_k_in = struct[0]
        if hasattr(self, 'q_loc') and self.q_loc: struct[0] = struct[-1]
        self.Layer_stack = nn.ModuleList([
            DecoderLayer(struct[i], v_out = struct[i+1], enc_attn_k_in = enc_attn_k_in, **kwargs)
            for i in range(len(struct)-1)])

    # q: former dec_output, k: last enc_output
    def forward(self, q, k):
        if q is None:
            q = k.clone()
        for Dec_layer in self.Layer_stack:
            q = Dec_layer(q, k)
        return q

class Transformer(Module, Impu_Module):
    def __init__(self, **kwargs):
        default = {'struct': [],
                   'n_head': 8,
                   'k_in': 0,
                   'k_out': '/8',
                   'fc': True,
                   'd_inner': '*4',
                   'lr': 1e-3,
                   'dropout': 0.1,
                   'dec_q': True,
                   'layer_norm': True}
        
        for key in default.keys():
            if key not in kwargs:
                kwargs[key] = default[key]
        
        self._name = 'Transformer'
        Module.__init__(self, **kwargs)
        if self.task == 'impu':
            Impu_Module.__init__(self, **kwargs)
     
        self.Encoder = Encoder(**kwargs)

        self.Decoder = Decoder(**kwargs)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 
        self.opt()

    def forward(self, x):
        enc_q = x
        if hasattr(self, 'dec_q') and self.dec_q: dec_q = self._nan * 1.0
        else: dec_q = None
        hid_z = self.Encoder(enc_q)
        dec_z = self.Decoder(dec_q, hid_z)
        # dec_z += enc_q
        return dec_z