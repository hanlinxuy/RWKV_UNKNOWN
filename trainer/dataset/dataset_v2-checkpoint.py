########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_lightning.utilities import rank_zero_info
# from .binidx import MMapIndexedDataset
# from .utils import MaybeIsPrime

instruction_path = ""
novel_path = ""

class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.data = np.load(args.data_file).astype("int")
        self.vocab_size = args.vocab_size
        self.data_size = len(self.data)
        self.my_indices = np.where(self.data[: -args.ctx_len-1 ] == 65456)[0].tolist()
        rank_zero_info("Current vocab size =", self.vocab_size, "(make sure it's correct)")
        rank_zero_info(f"Data has {self.data_size} tokens.")
        rank_zero_info(f"Data has {len(self.my_indices)} items.")

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size

        ctx_len = args.ctx_len
        req_len = ctx_len + 1
        magic_prime = args.magic_prime
        data = self.data

        # i = np.random.randint(0, self.data_size - req_len)
        i = random.choice(self.my_indices)

        dix = data[i : i + req_len]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y




class MyDatasetNew(Dataset):
    def __init__(self, args):
        self.args = args
        self.data = np.load(args.data_file).astype("int")
        self.vocab_size = args.vocab_size
        self.data_size = len(self.data)
        self.my_indices = np.where(self.data[: -args.ctx_len-1 ] == 65456)[0].tolist()
        rank_zero_info("Current vocab size =", self.vocab_size, "(make sure it's correct)")
        rank_zero_info(f"Data has {self.data_size} tokens.")
        rank_zero_info(f"Data has {len(self.my_indices)} items.")

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        rank = self.global_rank
        epoch = self.real_epoch
        world_size = self.world_size

        ctx_len = args.ctx_len
        req_len = ctx_len + 1
        magic_prime = args.magic_prime
        data = self.data

        # i = np.random.randint(0, self.data_size - req_len)
        i = random.choice(self.my_indices)

        dix = data[i : i + req_len]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
