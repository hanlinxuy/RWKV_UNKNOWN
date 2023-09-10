from task import TaskArgs

from trainer import trainpt
from dataset import load_data
from model import create_model


if __name__ == "__main__":
    args = TaskArgs()
    train_data, test_data_list = load_data(args)
    model = create_model(args)
    trainpt(args, model, train_data, test_data_list)
