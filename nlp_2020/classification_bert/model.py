import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel


class TextClassifier(nn.Module):
    """
    分类器
    """
    def __init__(
        self,
        output_dim=5,         # 输出维度，分类类别数，5分类
        dropout=0.1,
        bertConfig=None
    ):
        super().__init__()  # 调用父类构造函数初始化从父类继承的变量
        self.bertConfig = bertConfig
        # 构建bert层的时候需要使用bert的预训练模型的路径
        self.bert = BertModel.from_pretrained(bertConfig.bert_pretrain_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        # self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(bertConfig.hidden_size, output_dim)  # Bert输出，直接输出全连接层

    def forward(self, x):
        # x [ids, seq_len, mask]
        # 输入的x： (ids, seq_len, mask) ==>  list(list(int)), list(int), list(list(int))
        context = x[0]  # 对应输入的句子 shape[batch_size, pad_size] (32, 32)
        seq_len = x[1]
        mask = x[2]  # 对padding部分进行mask shape[128,32]
        # pooled层每次拿一句话的第一个单词输出作为dense层的输入， 因为self-attention不分先后
        # pooled: [batch_size, hidden_size]
        encoded_layers, pooled = self.bert(context, attention_mask=mask,
                                           output_all_encoded_layers=True)  # shape [128,768]
        # pooled = self.dropout(pooled)  # 每个句子
        out = self.linear(pooled)  # shape [batchsize,768]
        return out

