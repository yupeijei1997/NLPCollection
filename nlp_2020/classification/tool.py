import os
import logging
import jieba

import torch
import torch.nn as nn
from torchtext.data import Field, LabelField, TabularDataset

logger = logging.getLogger(__name__)


def build_and_cache_dataset(args, mode='train'):

    # TorchText采用声明式方法加载数据
    # 声明Field对象，这个Field对象指定你想要怎么处理某个数据
    # sequential序列化，use_vocab数字化，把单词映射成数字
    ID = Field(sequential=False, use_vocab=False)  # 不需要被序列化，不需要被数字化
    CATEGORY = LabelField(sequential=False, use_vocab=True, is_target=True)  # 不需要序列化，需要被数字化，是目标
    # tokenize传入一个函数，表示如何将文本字符串，切分成词或者字
    NEWS_TEXT = Field(
        sequential=True,  # 需要被序列化
        tokenize=jieba.lcut,  # 使用jieba分词切分
        include_lengths=True,  # 返回小型批处理的元组、长度列表
    )

    fields = [
        ('id', ID),
        (None, None),
        ('category', CATEGORY),
        ('news_text', NEWS_TEXT),
    ]

    logger.info("Creating features from dataset file at %s", args.data_dir)

    # Since dataset is split by `\t`.
    # 把train.csv中的每一行按\t做了划分，每一行生成一个Example对象
    dataset = TabularDataset(
        os.path.join(args.data_dir, f'{mode}.csv'),  # 数据集路径
        format='csv',  # 数据集格式
        fields=fields,  # 按何种方式处理数据，处理后每行数据符合'id': ，'category': ，'news_text': 的格式
        csv_reader_params={'delimiter': '\t'},  # 每行数据按\t划分，然后在按field定义的方式处理数据
    )

    features = ((ID, CATEGORY, NEWS_TEXT), dataset)
    return features


def save_model(args, model, optimizer, scheduler, global_step):
    # Save model checkpoint
    # 要保存的模型的路径与名称格式
    output_dir = os.path.join(args.output_dir, "ckpt-{}".format(global_step))
    # 不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Take care of distributed/parallel training

    # 在控制台打印信息
    logger.info("Saving model checkpoint to %s", output_dir)
    logger.info("Saving optimizer and scheduler states to %s", output_dir)

    # model.state_dict()里包含模型的结构和参数
    torch.save(model.state_dict(), os.path.join(output_dir, "model.pt"))
    # 优化器
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    # 学习率
    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))

