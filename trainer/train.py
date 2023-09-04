import os, warnings, math, datetime, sys, time
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)
import torch
from torch.utils.data import DataLoader
import deepspeed
from src.dataset_finetune import MyDataset
import types
from tqdm import tqdm
import random

def init_args(args):
    if args.random_seed >= 0:
        print(f"########## WARNING: GLOBAL SEED {args.random_seed} THIS WILL AFFECT MULTIGPU SAMPLING ##########\n" * 3)
        seed_everything(args.random_seed)

    if args.load_model == "":
        args.load_model = get_model(args.proj_dir)
    args.epoch_begin = next_model(args.proj_dir)
    args.accumulate_grad_batches = 4
    #args.max_epochs=1
    args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    args.enable_checkpointing = False
    args.replace_sampler_ddp = False
    args.logger = False
    #args.gradient_clip_val = 1.0
    #args.num_sanity_val_steps = 0
    #args.check_val_every_n_epoch = int(1e20)
    #args.log_every_n_steps = int(1e20)
    #args.max_epochs = -1  # continue forever
    #args.betas = (args.beta1, args.beta2)
    # args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
    # if args.dim_att <= 0:
    #     args.dim_att = args.n_embd
    # if args.dim_ffn <= 0:
    #     args.dim_ffn = args.n_embd * 4
    #args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"

    if not os.path.exists(args.proj_dir):
        os.makedirs(args.proj_dir)

    if "32" in args.precision:
        args.precision = 32
    elif args.precision == "fp16":
        args.precision = 16
    else:
        args.precision = "bf16"
    return args




if __name__ == "__main__":
    from argparse import ArgumentParser
    # from pytorch_lightning import Trainer
    # from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
    from utils import get_model,next_model


    args = types.SimpleNamespace()

    args.load_model = "../rwkv-output-models/3b-sft/rwkv-4.pth"
    args.wandb = ""
    args.proj_dir = "../rwkv-output-models/3b-sft"
    args.random_seed = -1

    args.data_file = "/home/neromous/sft"
    args.data_type = "utf-8"
    args.vocab_size = 65536

    args.ctx_len = 4096
    args.epoch_steps = 1000
    args.epoch_count = 500
    args.epoch_begin = 0
    args.epoch_save = 5
    args.micro_bsz = 1
    args.n_layer = 32
    args.n_embd =  2560
    #args.dim_att = 0
    #args.dim_ffn = 0
    args.pre_ffn = 0
    args.grad_cp = 1
    args.weight_decay = 0

    args.my_pile_version = 1
    args.my_pile_stage = 0
    args.my_pile_shift = -1
    args.my_pile_edecay = 0
    args.layerwise_lr = 1
    #args.ds_bucket_mb = 200
    # args.cuda_cleanup = 0

    #args.my_img_version = 0
    #args.my_img_size = 0
    #args.my_img_bit = 0
    #args.my_img_clip = 'x'
    #args.my_img_clip_scale = 1
    #args.my_img_l1_scale = 0
    #args.my_img_encoder = 'x'
    # args.my_img_noise_scale = 0
    #args.my_sample_len = 0
    #args.my_ffn_shift = 1
    #args.my_att_shift = 1
    args.my_pos_emb = 0
    #args.load_partial = 0
    #args.magic_prime = 0
    #args.my_qa_mask = 0
    #args.my_random_steps = 0
    args.my_testing = ''
    #args.my_exit = 99999999
    #args.my_exit_tokens = -1

    #
    args.num_nodes =  1
    args.devices = 1
    args.precision = "bf16"

    args.strategy =  "deepspeed_stage_2_offload"
    args =  init_args(args)

    os.environ["RWKV_T_MAX"] = str(args.ctx_len)
    os.environ["RWKV_MY_TESTING"] = args.my_testing
    os.environ["RWKV_FLOAT_MODE"] = args.precision
    os.environ["RWKV_JIT_ON"] = "1"
    from src.model import RWKV


    #########################################################################

    #samples_per_epoch = args.epoch_steps * args.real_bsz
    #tokens_per_epoch = samples_per_epoch * args.ctx_len
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    # allow tf32
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True


    ##################################################################################
    train_data = MyDataset(args)
    args.vocab_size = train_data.vocab_size
    model = RWKV(args)
    load_dict = torch.load(args.load_model, map_location="cpu")
    model.load_state_dict(load_dict)
    model_engine,optimizer,_,_ = deepspeed.initialize(model = model,
                                                      model_parameters=model.configure_optimizers(),
                                                      config="ds_config.config")


    for data in tqdm(train_data):
        # print("========",111)
        loss,logits,layer_logits = model_engine(data)
        model.zero_grad()
        model_engine.backward(loss)
        model_engine.step()
        # print("===loss===",loss)
        # print("===logits===",logits)
        # print("===layer logits[0]===",layer_logits[0])
