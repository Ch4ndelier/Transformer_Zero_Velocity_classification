import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

src_len = 24
tgt_len = 1

d_model = 6
d_ff = 24  # FeedForward dimension
d_k = d_v = 48  # dimension of Q(=K), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi_Head Attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # pe : [max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        # position : [max_len, 1]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x : [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q : [batch_size, n_heads, len_q, d_k]
        K : [batch_size, n_heads, len_k, d_k]
        V : [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask : [batch_size, n_heads, seq_len, seq_len]
        '''
        #  scores : [batch_size, n_heads, len_q, len_k]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)
        scores.masked_fill_(attn_mask, -1e9)
        #  attn : [batch_size, n_heads, len_q, len_k]
        attn = nn.Softmax(dim=-1)(scores)
        #  context : [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q : [batch_size, len_q, d_model]
        input_K : [batch_size, len_k, d_model]
        input_V : [batch_size, len_v(=len_k), d_model]
        TODO : len_q len_k len_v = src_len?
        attn_mask : [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        #  Q : [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  
        #  K : [batch_size, n_heads, len_k, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        #  V : [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)

        #  Q : [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        #  context : [batch_size, n_heads, len_q, d_v]
        #  attn : [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        #  context : [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)
        #  output : [batch_size, len_q, d_model]
        output = self.fc(context)
        return nn.LayerNorm(d_model).cuda()(output + residual), attn


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        residual = inputs
        output = self.fc(inputs)
        # return : [batch_size, seq_len, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.pos_emb = PositionalEncoding(d_model)


class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        # enc_inputs : [batch_size, src_len]
        # dec_inputs : [batch_size, tgt_len]

        # enc_outputs : [batch_size, src_len, d_model]
        # enc_self_attns : [n_layers, batch_size, n_heads, src_len, src_len]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outputs : [batch_size, tgt_len, d_model]
        # dec_self_attns : [n_layers, batch_size, n_heads, tgt_len, tgt_len]
        # dec_enc_attns : [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns