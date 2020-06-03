import torch
import torch.nn as nn
from pytorch_pretrained import BertModel, BertTokenizer


class BertConfig(object):
    """
    配置参数
    """
    def __init__(self):
        self.model_name = 'Bert'
        # epoch
        self.num_epochs = 10
        # batch_size
        self.batch_size = 37
        # 每句话处理的长度(短填，长切）
        self.pad_size = 32
        # 学习率
        self.learning_rate = 1e-5
        # bert预训练模型位置
        self.bert_pretrain_path = 'bert_pretrain'
        # bert切词器
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_pretrain_path)
        # bert隐层个数
        self.hidden_size = 768
        # 类别名
        self.class_list = ['news_agriculture', 'news_car', 'news_culture', 'news_edu', 'news_house']
        # 模型训练结果
        self.save_path = 'F:/nlp/shiyan/NLP2020-classification/model' + '/saved_dict' + self.model_name + '.ckpt'
        # 若超过1000bacth效果还没有提升，提前结束训练
        self.require_improvement = 1000
        # 日志
        self.log_path = 'F:/nlp/shiyan/NLP2020-classification' + '/log' + self.model_name

