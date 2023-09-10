__all__ = ["__create_model__"]
import importlib
import os
import sys

sys.path.append(os.path.dirname(__file__))


def create_model(args):
    if not args.model_base == "rwkv4":
        raise NotImplementedError
    os.environ["RWKV_T_MAX"] = str(args.ctx_len)
    os.environ["RWKV_MY_TESTING"] = "0"
    os.environ["RWKV_FLOAT_MODE"] = args.wkv_precision
    os.environ["RWKV_JIT_ON"] = "1"
    module = importlib.import_module(args.model_base)
    model = module.create_model(args)
    return model
