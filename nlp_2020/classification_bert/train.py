import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from nlp_2020.classification_bert.args import get_args
from nlp_2020.classification_bert.model import TextClassifier
from nlp_2020.classification_bert.bert.Bert import BertConfig
from nlp_2020.classification_bert import utils
from sklearn import metrics
from nlp_2020.classification_bert.pytorch_pretrained.optimization import BertAdam
import torch.nn.functional as F
import time
import numpy as np


def train(args):
    # 开始训练时间
    start_time = time.time()

    bertConfig = BertConfig()
    writer = SummaryWriter(log_dir=bertConfig.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))

    # 1.数据处理
    # bert切词器
    tokenizer = bertConfig.tokenizer
    # 划分后数据集每条数据的格式：(token_ids, label, seq_len, mask)
    # 对数据集进行划分
    train_data, dev_data, test_data = utils.bulid_dataset(
        dataset_path=r"F:\nlp\shiyan\NLP2020-classification\data\classification",  # 数据集路径
        tokenizer=tokenizer,  # 使用bert切词器处理text
        pad_size=bertConfig.pad_size)  # 规定句子长度为32

    # 构建训练集，验证集，测试集迭代器
    # 返回的每一条数据格式： (x, seq_len, mask), y
    train_iter = utils.bulid_iterator(train_data, batch_size=bertConfig.batch_size, device=args.device)
    dev_iter = utils.bulid_iterator(dev_data, batch_size=bertConfig.batch_size, device=args.device)
    test_iter = utils.bulid_iterator(test_data, batch_size=bertConfig.batch_size, device=args.device)

    # 2.实例化模型
    model = TextClassifier(
        output_dim=5,       # 输出的维度，即类别数
        dropout=args.dropout,
        bertConfig=bertConfig  # 参数
    )

    # optimizer, lr_scheduler, criterion
    model.to(args.device)

    param_optimizer = list(model.named_parameters())
    # 不需要衰减的参数
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    # 获取需要训练的参数
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay':0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_deacy':0.0}
    ]

    optimizer = BertAdam(params=optimizer_grouped_parameters,
                         lr=bertConfig.learning_rate,
                         warmup=0.05,
                         t_total=len(train_iter) * bertConfig.num_epochs)

    total_batch = 0  # 记录进行多少batch
    # 初始正无穷
    dev_best_loss = float('inf')  # 记录校验集合最好的loss
    last_imporve = 0  # 记录上次校验集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升，停止训练

    model.train()
    for epoch in range(bertConfig.num_epochs):
        print('Epoch [{}/{}'.format(epoch+1, bertConfig.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            # model.train()
            # 前向传播
            outputs = model(trains)
            # 梯度清零
            model.zero_grad()
            # 计算损失值
            loss = F.cross_entropy(outputs, labels)
            # 计算梯度
            loss.backward()
            # 更新参数
            optimizer.step()

            if total_batch % 100 == 0:  # 每多少次输出在训练集和校验集上的效果
                true = labels.data.cpu()
                # 取出每行最大值
                predit = torch.max(outputs.data, 1)[1].cpu()
                # 计算训练集准确率
                train_acc = metrics.accuracy_score(true, predit)
                # 评估
                # 计算验证集准确率与损失值
                dev_acc, dev_loss = evaluate(bertConfig, model, dev_iter)

                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    # 保存模型
                    torch.save(model.state_dict(), bertConfig.save_path)
                    improve = '*'
                    last_imporve = total_batch
                else:
                    improve = ''
                time_dif = utils.get_time_dif(start_time)
                msg = 'Iter:{0:>6}, Train Loss:{1:>5.2}, Train Acc:{2:>6.2}, Val Loss:{3:>5.2}, Val Acc:{4:>6.2%}, Time:{5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()

            total_batch = total_batch + 1
            if total_batch - last_imporve > bertConfig.require_improvement:
                # 在验证集合上loss超过1000batch没有下降，结束训练
                print('在校验数据集合上已经很长时间没有提升了，模型自动停止训练')
                flag = True
                break

        if flag:
            break
    writer.close()
    test(bertConfig, model, test_iter)


def evaluate(bertConfig, model, dev_iter, test=False):
    """

    :param config:
    :param model:
    :param dev_iter:
    :return:
    """
    model.eval()
    # 总损失值
    loss_total = 0
    # 验证集全部数据的预测值
    predict_all = np.array([], dtype=int)
    # 验证集全部数据的标签值
    labels_all = np.array([], dtype=int)

    # with里不反向传播
    with torch.no_grad():
        for texts, labels in dev_iter:
            # 前向传播
            outputs = model(texts)
            # 损失值
            loss = F.cross_entropy(outputs, labels)
            # 损失值累加
            loss_total = loss_total + loss
            # 标签值
            labels = labels.data.cpu().numpy()
            # 预测值
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            # 添加到全部标签里
            labels_all = np.append(labels_all, labels)
            # 添加到全部预测值里
            predict_all = np.append(predict_all, predict)

    # 对验证集所有数据计算准确率
    acc = metrics.accuracy_score(labels_all, predict_all)

    if test:
        # 精准率，召回率，f1_score
        report = metrics.classification_report(labels_all, predict_all, target_names=bertConfig.class_list, digits=4)
        # 混淆矩阵
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        # 测试集返回值
        return acc, loss_total / len(dev_iter), report, confusion

    # 验证集返回值
    return acc, loss_total / len(dev_iter)


def test(bertConfig, model, test_iter):
    """
    模型测试
    :param config:
    :param model:
    :param test_iter:
    :return:
    """
    model.load_state_dict(torch.load(bertConfig.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(bertConfig, model, test_iter, test=True)
    msg = 'Test Loss:{0:>5.2}, Test Acc:{1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score")
    print(test_report)
    print("Confusion Maxtrix")
    print(test_confusion)
    time_dif = utils.get_time_dif(start_time)
    print("使用时间：", time_dif)


def main():
    # get参数
    args = get_args()

    # 选择设备， 可在参数中指定是否使用cuda (no_cuda)
    device = "cuda" if torch.cuda.is_available() \
            and not args.no_cuda else "cpu"

    args.device = torch.device(device)      # 把device带入到args中，用来传递
    print("使用设备: ", device)

    # 正式开始训练
    train(args)


if __name__ == "__main__":
    main()


