from argparse import ArgumentParser
import deepspeed


class TaskArgs:
    def __init__(self):
        self.parsed = False
        parser = ArgumentParser("RWKV DEEPSPEED ONLY PROJECT")
        args = parser.parse_args()
        self.parsed = True

        self.setuplogger()
        self.dialect_alignment()

    def add_all_arguments(self, parser):
        parser = add_general_arguments(parser)
        return parser

    def setuplogger(self):
        pass

    def dialect_alignment(self):
        self.deepspeeed_dialect_alignment()

    def deepspeeed_dialect_alignment(self):
        pass

    def dump_config(self):
        pass

    def rank_zero_info(self):
        if self.global_rank == 0:
            self.logger.info()


def add_general_arguments(parser):
    parser.add_argument("--load_model", default="", type=str)  # full path, with .pth
    parser.add_argument("--wandb", default="", type=str)  # wandb project name. if "" then don't use wandb
    parser.add_argument("--proj_dir", default="out", type=str)
    parser.add_argument("--random_seed", default="-1", type=int)
    return parser


def add_deepspeed_arguments(parser):
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank passed from distributed launcher")
    parser = deepspeed.add_config_arguments(parser)
