import deepspeed
import torch
from tqdm import tqdm


def trainpt(args, model, train_data, test_data_list):
    if args.load_model != "":
        args.rank_zero_info("loading state dict")

        load_dict = torch.load(args.load_model, map_location="cpu")
        model.load_state_dict(load_dict)
    else:
        args.rank_zero_info("nothing, let it go")

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.configure_optimizers(),
        config=args.deepspeed_dialect,
    )

    for epoch in range(args.epoch_begin, args.epoch_count):
        __tqdm = tqdm(train_data, ncols=100)

        my_loss_count = 0
        for data in __tqdm:
            data = [x.to("cuda") for x in data]
            loss, logits, layer_logits = model_engine(data)
            my_loss_sum = loss.float().mean().item()
            my_loss_count += 1
            my_epoch_loss = my_loss_sum / my_loss_count

            model_engine.backward(loss)
            model_engine.step()
            __tqdm.set_description(f"epoch: {epoch+1}/{args.epoch_count} loss: {round(my_epoch_loss,4)} ")
