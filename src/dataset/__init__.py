__all__ = ["__load_data__"]
from .binidxdata import BinIdxData


def load_data(args):
    train_dataset = None
    test_datasets = []
    if args.data_type == "binidx":
        train_dataset = BinIdxData(args)
    else:
        raise NotImplementedError

    return train_dataset, test_datasets
