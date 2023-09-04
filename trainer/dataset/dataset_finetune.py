import json, math, random, os, sys
import numpy as np
import torch
from torch.utils.data import Dataset
from .dataloader import dataloader
import random
import copy

class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.vocab_size = args.vocab_size
        self.data = dataloader(args.data_file)
        self.pool = self.limit_sample(args.epoch_steps)
        self.data_size = len(self.data)
        self.item = []
 
    def __len__(self):
        return self.args.epoch_steps * self.args.micro_bsz

    def limit_sample(self,n):
        if len(self.data) <= n:
            res = random.sample(self.data, len(self.data))
        else:
            res = random.sample(self.data, n)
        return res

    def __getitem__(self, idx):
        args = self.args
        ctx_len = args.ctx_len
        req_len = ctx_len + 1
        # 训练数据构建
        if len(self.item) == 0:
            if len(self.pool) == 0:
                self.pool = self.limit_sample(args.epoch_steps)
            self.item = self.pool[0]
            self.pool = self.pool[1:]
        step  = self.item[:req_len]
        step_len =  len(step)
        # 滑窗机制
        if len(self.item) > req_len:
            half = int(ctx_len / 2)
            self.item = self.item[half:]
        else:
            self.item = self.item[req_len:]
        dix = [0 for x in range(req_len)]
        dix[:step_len] = step
        # 生成mask
        mask = [int(x!=0) for x in dix]
        mask = mask[:-1]
        # 输入构建
        x = torch.tensor([dix[:-1]], dtype=torch.long).to('cuda')
        y = torch.tensor([dix[1:]], dtype=torch.long).to('cuda')
        z = torch.tensor([mask], dtype=torch.float32).to('cuda')
        return x, y, z
