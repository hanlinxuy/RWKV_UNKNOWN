from .dsmodel import RWKV
import os


def create_model(args):
    model = RWKV(args)
    return model
