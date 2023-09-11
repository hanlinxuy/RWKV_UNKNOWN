__all__ = ["__BinIdxData__"]
from .dataset import MyDataset
from torch.utils.data import DataLoader


def BinIdxData(args):
    dataset = MyDataset(args)
    data_loader = DataLoader(
        dataset,
        shuffle=False,
        pin_memory=True,
        batch_size=args.micro_bsz,
        num_workers=1,
        persistent_workers=False,
        drop_last=True,
    )
    return data_loader
