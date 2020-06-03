# pylint: disable=bad-continuation
import argparse


def get_args():
    # 创建ArgumentParser()对象
    parser = argparse.ArgumentParser()

    # WORKSPACE = "E:/Workspaces/Python/NLP/NLP2020"
    WORKSPACE = r"F:\nlp\shiyan\NLP2020-classification"
    DATADIR = f"{WORKSPACE}/data"
    data_dir = f"{DATADIR}/classification"
    model_name_or_path = f"{DATADIR}/model"
    output_dir = f"{DATADIR}/output"
    cache_dir = f"{DATADIR}/cache"
    # embed_path = f"{DATADIR}/sgns.sogounews.bigram-char"
    embed_path = f"{DATADIR}/sgns.renmin.bigram-char"

    # 使用add_argument()方法，添加参数
    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=data_dir,
        type=str,
        # required=True,
        help="The input data dir.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=model_name_or_path,
        type=str,
        # required=True,
        help="Path to pre-trained model or shortcut name.",
    )
    parser.add_argument(
        "--output_dir",
        default=output_dir,
        type=str,
        # required=True,
        help="The output directory where the model predictions "
        "and checkpoints will be written.",
    )
    parser.add_argument(
        "--embed_path",
        default=embed_path,
        type=str,
        # required=True,
        help="Embedding path",
    )
    parser.add_argument(
        "--cache_dir",
        default=cache_dir,
        type=str,
        help="Cache data binary data or other files",
    )

    # Other parameters
    parser.add_argument("--train_batch_size",
                        default=128,
                        type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size",
                        default=64,
                        type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    # Model config
    parser.add_argument("--num_labels",
                        default=5,
                        type=int,
                        help="Number of labels")
    parser.add_argument("--vocab_size",
                        default=400000,
                        type=int,
                        help="Size of vocabulary")
    parser.add_argument("--dropout",
                        default=0.1,
                        type=float,
                        help="dropout prob")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm",
                        default=1.0,
                        type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs",
                        default=10,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps",
                        type=int,
                        default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps",
                        type=int,
                        default=100,
                        help="Save checkpoint every X updates steps.")
    # default参数不写的话，默认值都为False
    parser.add_argument("--overwrite_output_dir",
                        action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--no_cuda",
                        action="store_true",
                        help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed",
                        type=int,
                        default=66,
                        help="random seed for initialization")

    # 使用parse_args()方法，解析添加的参数，存在args里
    args = parser.parse_args()
    return args
