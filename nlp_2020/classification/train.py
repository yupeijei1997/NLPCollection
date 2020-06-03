import os
import logging
from typing import List, Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext.vocab import Vectors
from torchtext.data import BucketIterator
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from sklearn.metrics import precision_recall_fscore_support

from nlp_2020.classification.args import get_args
from nlp_2020.classification.model import TextClassifier
from nlp_2020.classification.tool import build_and_cache_dataset, save_model

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def train(args, writer):

    # 1.数据处理
    # 获得预定义的fields，划分过的训练数据集
    # train_dataset中的每一行是一个torchtext.data.Example对象，这个对象的'id': ，'category': ，'news_text': 这三个属性保存了原来csv中每一行的数据
    # 此时还未数字化，要等到构造迭代器的时候才数字化
    fields, train_dataset = build_and_cache_dataset(args, mode='train')

    # NEWS_TEXT，CATEGORY是要存词汇表的，之后构造迭代器的时候会用上
    ID, CATEGORY, NEWS_TEXT = fields
    # 词向量
    vectors = Vectors(name=args.embed_path, cache=args.data_dir)

    # import gensim
    # word2vec = gensim.models.KeyedVectors.load_word2vec_format(args.embed_path, binary=True)

    # 创建数据集的词汇表，同时加载预训练的词向量
    # 创建词汇表，作为一个Vocab对象，存在Field对象NEWS_TEXT里，其中stoi是词和数字的映射字典，vectors是词的词向量矩阵，两者是对应的，第一个词映射为0，且词向量在vectors里也是第一行
    NEWS_TEXT.build_vocab(
        train_dataset,  # 根据训练数据集创建词汇表
        max_size=args.vocab_size,  # 句子最大长度
        vectors=vectors,  # 根据词汇表，从加载的预训练词向量中抽出相应的词向量
        unk_init=torch.nn.init.xavier_normal_,
    )
    # 创建标签的词汇表，作为一个Vocab对象，存在Field对象CATEGORY里
    CATEGORY.build_vocab(train_dataset)
    # 实例化模型
    model = TextClassifier(
        vocab_size=len(NEWS_TEXT.vocab),  # 训练集划分后的词的总个数，即词汇表长度
        output_dim=args.num_labels,  # 类别数
        pad_idx=NEWS_TEXT.vocab.stoi[NEWS_TEXT.pad_token],  # NEWS_TEXT.pad_token = <pad>，从stoi（'<pad> : 1'）里取出<pad>的值
        dropout=args.dropout,
    )

    # 为embedding层的矩阵赋值为NEWS.vocab.vectors
    model.embedding.from_pretrained(NEWS_TEXT.vocab.vectors)

    # 构造训练集迭代器，在这一步将torchtext.data.Example对象中的news_text属性数字化
    # 还会对同一个batch内的不够长的句子做pad，pad成batch内最长的句子的长度，但是在batch.news_text里会记录句子真实的长度
    bucket_iterator = BucketIterator(
        train_dataset,
        batch_size=args.train_batch_size,  # batch_size大小
        sort_within_batch=True,  # batch内排序
        shuffle=True,  # 2.batch间进行乱序
        sort_key=lambda x: len(x.news_text),  # 1.按句子长度排序，x代表训练集中的每一行,即一个torchtext.data.Example对象
        device=args.device,  # 放入GPU里
    )

    # 2.训练
    model.to(args.device)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    # 优化器
    optimizer = Adam(model.parameters(),
                     lr=args.learning_rate,
                     eps=args.adam_epsilon)
    # 学习率随epoch改变
    scheduler = OneCycleLR(optimizer,
                           max_lr=args.learning_rate * 10,
                           epochs=args.num_train_epochs,
                           steps_per_epoch=len(bucket_iterator))

    global_step = 0
    # 梯度清零
    model.zero_grad()

    # tqdm(list) 方法可以传入任意一种list

    # trange(i) 是 tqdm(range(i)) 的简单写法
    # 下式左边，等价于tqdm(range(0, 5))
    train_trange = trange(0, args.num_train_epochs, desc="Train epoch")

    for _ in train_trange:
        epoch_iterator = tqdm(bucket_iterator, desc='Training')  # 进度条

        # 对每个batch做一个前向传播和反向传播，更新参数
        for step, batch in enumerate(epoch_iterator):  # for循环结束进度条才为100%
            model.train()

            # news_text：所有句子组成一个list[[句子1]，[句子2]，...]，实际是按列是一个句子
            # [句子1] = [单词1(单词对应的下标)，单词2，单词3,...]
            # news_text_lengths：所有句子的长度组成一个list
            news_text, news_text_lengths = batch.news_text  # news_text中，每一列是一个数字化后的句子，batch_size是多少，就有多少列
            # print(batch.news_text)
            #
            # print(len(news_text))
            # print(news_text.shape)
            #
            # print(len(news_text_lengths))
            # print(news_text_lengths)
            category = batch.category  # 标签的list

            # 前向传播
            preds = model(news_text, news_text_lengths)

            # 计算损失值
            loss = criterion(preds, category)
            # 计算梯度
            loss.backward()

            # loss随每次batch的变化，写入tensorboard
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            # 学习率随每次batch的变化，写入tensorboard
            writer.add_scalar('Train/lr',
                              scheduler.get_last_lr()[0], global_step)

            # NOTE: Update model, optimizer should update before scheduler
            # 更新参数
            optimizer.step()
            # 更新学习率
            scheduler.step()
            # 记录用过多少个batch进行参数更新了
            global_step += 1

            # 评估
            # 每50轮评估一次
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                # 返回损失值，精准率，召回率，f1_score的字典
                results = evaluate(args, model, CATEGORY.vocab, NEWS_TEXT.vocab)

                # 损失值，精准率，召回率，f1_score随每次batch的变化，写入tensorboard
                for key, value in results.items():
                    writer.add_scalar("Eval/{}".format(key), value,
                                      global_step)

            # 每100轮保存一次模型
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                save_model(args, model, optimizer, scheduler, global_step)

    writer.close()


def evaluate(args, model, category_vocab, example_vocab, mode='dev'):

    # 获得预定义的fields，划分过的验证数据集
    fields, eval_dataset = build_and_cache_dataset(args, mode=mode)
    # 构造验证集迭代器
    bucket_iterator = BucketIterator(
        eval_dataset,
        train=False,
        batch_size=args.eval_batch_size,
        sort_within_batch=True,
        sort_key=lambda x: len(x.news_text),
        device=args.device,
    )
    ID, CATEGORY, NEWS_TEXT = fields
    # 标签词汇表
    CATEGORY.vocab = category_vocab
    # 训练集数据词汇表
    NEWS_TEXT.vocab = example_vocab
    # 在控制台打印信息
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # 评估
    model.eval()
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    eval_loss, eval_steps = 0.0, 0
    labels_list, preds_list = [], []

    for batch in tqdm(bucket_iterator, desc='Evaluation'):
        news_text, news_text_lengths = batch.news_text  # 数字化后的验证集数据的list
        labels = batch.category  # 标签的list
        with torch.no_grad():
            # 前向传播
            logits = model(news_text, news_text_lengths)
            # 损失值
            loss = criterion(logits, labels)
            # 累加损失值
            eval_loss += loss.item()

        # 记录进行了第几轮用batch更新参数
        eval_steps += 1
        # 预测值
        preds = torch.argmax(logits, dim=1)
        # 预测值list
        preds_list.append(preds)
        # 标签list
        labels_list.append(labels)

    # torch.cat()将list内的张量拼接在一起，成为一个Tensor
    # .detach().cpu().numpy()取出Tensor中的值，全部放在一个list里
    y_true = torch.cat(labels_list).detach().cpu().numpy()
    y_pred = torch.cat(preds_list).detach().cpu().numpy()

    # 计算精准率、召回率、f1_score
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro')

    # Write into tensorboard
    # TODO: recore false-pos and false-neg samples.

    # 将损失值和准确率存成一个字典
    results = {
        'loss': eval_loss / eval_steps,
        'f1': f1_score,
        'precision': precision,
        'recall': recall,
    }
    # 在控制台打印，损失值，精准率，召回率，f1_score
    msg = f'*** Eval: loss {loss}, f1 {f1_score}, precision {precision}, recall {recall}'
    logger.info(msg)

    return results


def main():
    args = get_args()
    writer = SummaryWriter()

    # Check output dir
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # os.listdir()返回指定的文件夹，包含的文件和文件夹的名字的列表
    # if not ...，可以理解为不为true的话
    if os.path.exists(args.output_dir) \
            and os.listdir(args.output_dir) \
            and not args.overwrite_output_dir:
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome.")

    # Set device
    device = "cuda" if torch.cuda.is_available() \
            and not args.no_cuda else "cpu"
    # print(torch.device(device))

    # torch.device(device)代表将torch.Tensor分配到的设备的对象
    args.device = torch.device(device)
    # print(args.device)
    # 在终端输出信息 - INFO - __main__ - Process device: cuda
    logger.info("Process device: %s", device)

    train(args, writer)


if __name__ == "__main__":
    main()
