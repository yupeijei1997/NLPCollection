import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class TextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,  # 词汇表长度
        output_dim,  # 输出维度，可看作类别数
        n_layers=2,  # 2层LSTM
        pad_idx=None,
        hidden_dim=128,  # LSTM层隐藏单元个数
        embed_dim=300,  # 词向量维度
        dropout=0.5,
        bidirectional=True,  # 双向LSTM
    ):
        super().__init__()
        num_directions = 1 if not bidirectional else 2  # 双向就为2，单向为1

        self.embedding = nn.Embedding(
            vocab_size,  # 词汇表的长度
            embed_dim,  # 词向量的维度
            padding_idx=pad_idx,
        )
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(hidden_dim * n_layers * num_directions, output_dim)
        # self.linear = nn.Linear(32 * hidden_dim * num_directions,
        #                         output_dim)

    def forward(self, x, x_len):
        # 将每个词转换为对应的300维词向量
        x = self.embedding(x)
        # Pad each sentences for a batch,
        # the final x with shape (seq_len, batch_size, embed_size)

        # 第一话句话x，x1的词向量，x2的词向量，第二句话y，y1的词向量，y2的词向量，...，连起来成为x
        # 把batch_size个数字化后的数据传给LSTM层
        x = pack_padded_sequence(x, x_len)  # x_len每句话的长度

        # h_n: (num_layers * num_directions, batch_size, hidden_size)
        # NOTE: take the last hidden state of encoder as in seq2seq architecture.

        # 每次把一个单词的词向量输入给LSTM，按x1的词向量，x2的词向量，...，y1的词向量，y2的词向量，...这样的顺序
        hidden_states, (h_n, c_c) = self.lstm(x)  # LSTM接收的输入是300维的词向量

        h_n = torch.transpose(self.dropout(h_n), 0, 1).contiguous()
        # h_n:(batch_size, hidden_size * num_layers * num_directions)
        h_n = h_n.view(h_n.shape[0], -1)
        loggits = self.linear(h_n)

        # hidden_states = nn.utils.rnn.pad_packed_sequence(hidden_states, batch_first=False)[0]
        # hidden_states = torch.transpose(self.dropout(hidden_states), 0, 1).contiguous()
        # # h_n:(batch_size, hidden_size * num_layers * num_directions)
        # hidden_states = hidden_states.view(hidden_states.shape[0], -1)
        # loggits = self.linear(hidden_states)

        return loggits

