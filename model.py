
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class sqrtReLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.act =  nn.ReLU()
    def forward(self, x):
        return torch.sqrt(self.act(x))


class AutoEncoder(nn.Module):
    def __init__(self, num_points, hidden_ch = 1):
        super(AutoEncoder, self).__init__()
        self.num_points = num_points

        self.encoder = Encoder(num_points, hidden_ch)
        self.decoder = Decoder(num_points, hidden_ch)

    def forward(self, x):
        embedding = self.encoder(x)
        x = self.decoder(embedding)
        return x, embedding



class Encoder(nn.Module):
    def __init__(self, num_points, hidden_ch = 1, num_hidden_layers = 1):
        super(Encoder, self).__init__()
        self.num_points = num_points
        self.act = sqrtReLU()
        self.first_layer = nn.Sequential(nn.Linear(num_points*3, hidden_ch, bias=True), self.act)


    def forward(self, x):
        x=x.view(-1, self.num_points*3)
        x = self.first_layer(x)

        return x


class Decoder(nn.Module):
    def __init__(self, num_points, hidden_ch = 1, dtype=torch.float32):
        super(Decoder, self).__init__()
        self.dtype = dtype
        self.num_points = num_points
        self.last_layer = nn.Linear(1, num_points*3, bias=True, dtype=dtype)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.last_layer(x)
        x = self.sigmoid(x)
        x=x.view(-1, 3, self.num_points)

        return x
