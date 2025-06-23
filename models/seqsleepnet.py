# -*- coding:utf-8 -*-

import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(
            self, input_size, hidden_size, num_layers,
            dropout=0.5, bidirectional=False, return_last=True
        ):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.return_last = return_last
        self.D = 1 
        if bidirectional is True: self.D = 2
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=bidirectional
            )
    
    def forward(self, x, h0=None, c0=None):
        if h0 is None:
            h0 = torch.zeros(self.num_layers*self.D, x.size(0), self.hidden_size).to(x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers*self.D, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # RNN output (batch_size, seq_length, hidden_size)
        if self.return_last: out = out[:, -1, :]
        return out


class LSTM_Attention(nn.Module):
    def __init__(
            self, input_size, hidden_size, num_layers, attention_dim=64,
            dropout=0.5, bidirectional=False,
        ):
        super(LSTM_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.D = 1 
        if bidirectional is True: self.D = 2
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, 
            batch_first=True, dropout=dropout, bidirectional=bidirectional
        )
        self.attn_pooling = nn.Sequential(
            nn.Linear(self.D*hidden_size, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x, h0=None, c0=None):
        if h0 is None:
            h0 = torch.zeros(self.num_layers*self.D, x.size(0), self.hidden_size).to(x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers*self.D, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))  # RNN output (batch_size, seq_length, hidden_size)
        attn_w = self.attn_pooling(out)
        out = torch.sum(out * attn_w.expand_as(out), dim=1)
        return out


class SeqSleepNet(nn.Module):
    def __init__(
            self, encoder, num_classes=0, n_seqlen=300, dropout=0.5,
            num_layers=2, hidden_dim=128, bidirectional=False,
            freeze_encoder=False,
        ):
        super().__init__()
        self.n_seqlen = n_seqlen

        self.encoder = encoder
        encoder_dim = encoder.feature_dim
        if freeze_encoder: # freeze the encoder
            self.freeze_encoder(True)

        self.seq_encoder = LSTM(encoder_dim, hidden_dim, num_layers, bidirectional=bidirectional)
        self.drop = nn.Dropout(dropout)
        # classfier head
        rnn_outdim = 2 * hidden_dim if bidirectional else hidden_dim
        self.head = nn.Linear(rnn_outdim, num_classes) if num_classes > 0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.xavier_uniform_(m.weight, gain=1)
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def freeze_encoder(self, freeze=True):
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            for param in self.encoder.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = x.reshape((-1,)+ x.shape[2:]) # (batch * seqlen, n_timepoints, n_channels)
        # embed patches
        x = self.encoder(x)
        x_seq = x.view(-1, self.n_seqlen, x.size(-1)) # (batch, seqlen, hidden)
        x = self.seq_encoder(x_seq)
        x = self.drop(x)
        return self.head(x)


if __name__ == '__main__':

    from sleepnet import TinySleepNet, DeepSleepNet

    x = torch.randn((20, 10, 1, 3000, 1))

    base_encoder = DeepSleepNet(0, 3000)
    # base_encoder = TinySleepNet(0, 3000)

    model = SeqSleepNet(
        base_encoder, 
        num_classes=5,
        n_seqlen=10,
    )
    print(model)
    y = model(x)
    print(y.shape)