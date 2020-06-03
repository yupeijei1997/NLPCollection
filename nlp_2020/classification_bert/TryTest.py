from nlp_2020.classification_bert.bert.Bert import BertConfig
from nlp_2020.classification_bert import utils
from nlp_2020.classification_bert.args import get_args
import torch

args = get_args()
device = "cuda"

args.device = torch.device(device)  # 把device带入到args中，用来传递
bertConfig = BertConfig()

#### 1. 数据集
tokenizer = bertConfig.tokenizer
# (token_ids, label, seq_len, mask)
train_data, dev_data, test_data = utils.bulid_dataset(
    dataset_path=r"F:\nlp\shiyan\NLP2020-classification\data\classification",
    tokenizer=tokenizer,
    pad_size=bertConfig.pad_size)

# 返回的每一条数据格式： (x, seq_len, mask), y
train_iter = utils.bulid_iterator(train_data, batch_size=bertConfig.batch_size, device=args.device)
# dev_iter = utils.bulid_iterator(dev_data, batch_size=bertConfig.batch_size, device=args.device)
# test_iter = utils.bulid_iterator(test_data, batch_size=bertConfig.batch_size, device=args.device)

index = 1
for item in train_iter:
    x, y = item

    # if index == 626:
    print()
    print(index, len(x[0]), len(y))
    index += 1

