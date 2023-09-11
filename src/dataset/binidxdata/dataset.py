########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from .binidx import MMapIndexedDataset
from .utils import MaybeIsPrime


class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args

        self.vocab_size = args.vocab_size
        args.rank_zero_info(f"Current vocab size = {self.vocab_size} (make sure it's correct)")

        self.data = MMapIndexedDataset(args.data_file)
        self.data_size = len(self.data._bin_buffer) // self.data._index._dtype_size
        args.rank_zero_info(f"Data has {self.data_size} tokens.")

    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def __getitem__(self, idx):
        args = self.args
        # rank = self.global_rank
        # epoch = self.real_epoch
        # world_size = self.world_size
        args.rank_zero_debug(f"epoch {epoch} idx {idx} rank {rank}/{world_size}")
        ctx_len = args.ctx_len
        req_len = ctx_len + 1
        # magic_prime = args.magic_prime
        data = self.data

        # cheat: pick a random spot in dataset
        i = np.random.randint(0, self.data_size - req_len)
        dix = data.get(idx=0, offset=i, length=req_len).astype(int)

        input_ids = torch.tensor(dix[:-1], dtype=torch.long)
        targets = torch.tensor(dix[1:], dtype=torch.long)
        mask = torch.ones(targets.shape, dtype=torch.bfloat16)
        return input_ids, targets, mask
