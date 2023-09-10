from argparse import ArgumentParser
import deepspeed
import collections
from .set_seed import setup_seed
import logging
import os
import json

base_path = f"{os.path.dirname(__file__)}/../../"


class TaskArgs:
    def __init__(self):
        self.name = "TaskArgs"
        self.parsed = False
        self.global_rank = int(os.getenv("RANK", 0))
        parser = ArgumentParser("RWKV DEEPSPEED ONLY PROJECT")
        parser = self.add_all_arguments(parser)
        args = parser.parse_args()
        self.parsed = True

        for key in vars(args):
            setattr(self, key, getattr(args, key))

        setup_seed(self)
        self.setuplogger()
        self.dialect_alignment()

    def add_all_arguments(self, parser):
        parser = add_general_arguments(parser)
        parser = add_training_arguments(parser)
        parser = add_model_arguments(parser)
        parser = add_data_arguments(parser)
        parser = add_deepspeed_arguments(parser)
        return parser

    def setuplogger(self):
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(self.loglevel)
        handler = logging.StreamHandler()
        formatter = logging.Formatter("| %(asctime)s | %(filename)s:%(lineno)d | %(levelname)s: %(message)s |")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def dialect_alignment(self):
        self.deepspeeed_dialect_alignment()

    def deepspeeed_dialect_alignment(self):
        self.deepspeed_dialect = collections.defaultdict(list)
        if self.deepspeed:
            self.rank_zero_info("using deepspeed config first")
        else:
            self.rank_zero_info("using other config to build deepspeed config")
            raise NotImplementedError

        with open(self.deepspeed_config, "r") as config_file:
            self.deepspeed_dialect = json.load(config_file)
            
        self.deepspeed_dialect["train_micro_batch_size_per_gpu"] = self.micro_bsz
        # self.deepspeed_dialect["train_micro_batch_size_per_gpu"] = self.micro_bsz
        
    def dump_config(self):
        pass

    def rank_zero_info(self, string):
        if self.global_rank == 0:
            self.logger.info(string)

    def rank_zero_debug(self, string):
        if self.global_rank == 0:
            self.logger.debug(string)


def add_general_arguments(parser):
    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="1", type=int)
    parser.add_argument("--loglevel", default="INFO", type=str)
    return parser


def add_training_arguments(parser):
    parser.add_argument("--ctx_len", default=1024, type=int)
    parser.add_argument("--epoch_steps", default=1000, type=int)  # a mini "epoch" has [epoch_steps] steps
    parser.add_argument(
        "--epoch_count", default=500, type=int
    )  # train for this many "epochs". will continue afterwards with lr = lr_final
    parser.add_argument(
        "--epoch_begin", default=0, type=int
    )  # if you load a model trained for x "epochs", set epoch_begin = x
    parser.add_argument("--epoch_save", default=5, type=int)  # save the model every [epoch_save] "epochs"
    parser.add_argument("--micro_bsz", default=12, type=int)  # micro batch size (batch size per GPU)

    parser.add_argument("--lr_init", default=1e-4, type=float)
    parser.add_argument("--lr_final", default=1e-5, type=float)
    parser.add_argument("--warmup_steps", default=-1, type=int)  # try 50 if you load a model
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.99, type=float)  # use 0.999 when your model is close to convergence
    parser.add_argument("--adam_eps", default=1e-8, type=float)
    parser.add_argument("--grad_cp", default=0, type=int)  # gradient checkpt: saves VRAM, but slower
    parser.add_argument("--dropout", default=0, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)  # try 0.1 / 0.01 / 0.001
    parser.add_argument("--wkv_precision", default="bf16", type=str)
    parser.add_argument("--layerwise_lr", default=1, type=int)  # layerwise lr for faster convergence (but slower it/s)
    return parser


def add_model_arguments(parser):
    parser.add_argument("--model_base", default="rwkv4", type=str)
    parser.add_argument("--model_arch", default="base", type=str)
    parser.add_argument("--n_layer", default=6, type=int)
    parser.add_argument("--n_embd", default=512, type=int)
    # below are now stable arguments, ignore first
    parser.add_argument("--dim_att", default=0, type=int)
    parser.add_argument("--dim_ffn", default=0, type=int)
    parser.add_argument("--pre_ffn", default=0, type=int)  # replace first att layer by ffn (sometimes better)
    parser.add_argument("--head_qk", default=0, type=int)  # my headQK trick
    parser.add_argument("--tiny_att_dim", default=0, type=int)  # tiny attention dim
    parser.add_argument("--tiny_att_layer", default=-999, type=int)  # tiny attention @ which layer
    return parser


def add_data_arguments(parser):
    # data space should include tokenizer, but you can decide if you want to use
    parser.add_argument("--data_file", default=base_path + "/example_inputs/dollytest_text_document", type=str)
    parser.add_argument("--data_type", default="binidx", type=str)
    parser.add_argument(
        "--vocab_size", default=65536, type=int
    )  # vocab_size = 0 means auto (for char-level LM and .txt data)
    return parser


def add_deepspeed_arguments(parser):
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank passed from distributed launcher")
    parser = deepspeed.add_config_arguments(parser)
    return parser
