import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
# from time_dataset import TimeSeriesDataset
from torch.nn import functional as F
import math


class Input(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.linear = nn.Linear(801, num_hidden)

    def forward(self, x):
        output = self.linear(x)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, num_hidden, dropout=0.1, len=11):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        PE = torch.zeros(len, num_hidden)
        position = torch.arange(0, len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, num_hidden, 2).float() * (-math.log(10000.0) / num_hidden))
        PE[:, 0::2] = torch.sin(position * div_term)
        PE[:, 1::2] = torch.cos(position * div_term)
        PE = PE.unsqueeze(0).transpose(0, 1)

        self.register_buffer('PE', PE)

    def forward(self, X):
        X = X + self.PE[:X.size(0), :]
        return X


class FFN(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.dense1 = nn.Linear(num_hidden, num_hidden*2)
        self.dense2 = nn.Linear(num_hidden*2, num_hidden)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.dense2(self.relu(self.dense1(x)))
        return output


class AddNorm(nn.Module):
    def __init__(self, num_hidden, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalize = nn.LayerNorm(num_hidden)

    def forward(self, X, Y):
        Y = self.dropout(Y) + X
        return self.normalize(Y)


class MultiHeadAttention(nn.Module):
    def __init__(self, num_hidden, num_head, dropout):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_heads = num_head
        self.dropout = nn.Dropout(dropout)
        self.normalize = AddNorm(num_hidden, dropout)

        # define q,k,v linear layer
        self.Wq = nn.Linear(self.num_hidden, self.num_hidden)
        self.Wk = nn.Linear(self.num_hidden, self.num_hidden)
        self.Wv = nn.Linear(self.num_hidden, self.num_hidden)

        self.relu = nn.ReLU()
        self.Q = nn.Sequential(self.Wq, self.relu)
        self.K = nn.Sequential(self.Wk, self.relu)
        self.V = nn.Sequential(self.Wv, self.relu)

    def forward(self, queries, keys, values):
        # get matrices of q, k, v
        q, k, v = self.Q(queries), self.K(keys), self.V(values)
        # 得到经多头切分后的矩阵 shape:(batch_size, len_seq, d_model/num_heads)[在最后一维切分]
        q_split = torch.chunk(q, self.num_heads, dim=-1)
        k_split = torch.chunk(k, self.num_heads, dim=-1)
        v_split = torch.chunk(v, self.num_heads, dim=-1)
        # 将他们在第二维（new）堆叠  shape:(batch_size, num_heads, len_seq, d_model/num_heads)
        q_stack = torch.stack(q_split, dim=1)
        k_stack = torch.stack(k_split, dim=1)
        v_stack = torch.stack(v_split, dim=1)
        # get attention score
        score = torch.matmul(q_stack, k_stack.permute(0, 1, 3, 2))
        score = score / (k_stack.size()[-1] ** 0.5)

        score = F.softmax(score, dim=-1)
        a = torch.matmul(score, v_stack)
        a = torch.reshape(a.permute(0, 1, 3, 2), shape=(q.size(0), q.size(1), q.size(2)))
        a += queries

        return a


class EncoderBlock(nn.Module):
    def __init__(self, num_hidden, num_head, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(num_hidden=num_hidden,
                                       num_head=num_head,
                                       dropout=dropout)
        self.addnorm1 = AddNorm(num_hidden, dropout)
        self.addnorm2 = AddNorm(num_hidden, dropout)
        self.FFN = FFN(num_hidden=num_hidden)

    def forward(self, X):
        Y = self.addnorm1(X, self.attn(X, X, X))
        outputs = self.addnorm2(Y, self.FFN(Y))
        return outputs


class TimeEncoder(nn.Module):
    def __init__(self, num_hidden, num_head, num_blocks, dropout):
        super().__init__()
        self.pe = PositionalEncoding(num_hidden=num_hidden)
        self.layers = nn.ModuleList([EncoderBlock(num_hidden=num_hidden,
                                                  num_head=num_head,
                                                  dropout=dropout) for _ in range(num_blocks)])
        self.input_layer = Input(num_hidden=num_hidden)

        # self.projection = ClassifyHeadTime()

    def forward(self, x):
        input = self.input_layer(x)
        enc_output = self.pe(input.transpose(0, 1)).transpose(0, 1)
        for layer in self.layers:
            enc_output = layer(enc_output)

        # output = enc_output.view(enc_output.size(0), -1)
        # output = self.projection(output)

        return enc_output


class ClassifyHeadTime(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(11264, 2048),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(2048),
                                    nn.Dropout(0.1),
                                    nn.Linear(2048, 512))
        self.dense2 = nn.Sequential(
                                   nn.ReLU(),
                                   nn.BatchNorm1d(512),
                                   nn.Dropout(0.1),
                                   nn.Linear(512, 128),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(128),
                                   nn.Dropout(0.1),
                                   nn.Linear(128, 1))

    def forward(self, x):
        output = self.dense2(self.dense1(x))
        return output


class ClassifyHeadHELM(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense1 = nn.Sequential(nn.Linear(18432, 2048),
                                    nn.ReLU(),
                                    nn.BatchNorm1d(2048),
                                    nn.Dropout(0.1),
                                    nn.Linear(2048, 512))
        self.dense2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            nn.Linear(128, 1))

    def forward(self, x):
        output = self.dense2(self.dense1(x))
        return output












