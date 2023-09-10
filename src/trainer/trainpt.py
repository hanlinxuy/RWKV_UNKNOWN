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
        __tqdm = tqdm(len(train_data), ncols=50)
        __tqdm.set_description(f"epoch: {epoch+1}/{args.epoch_count}")
        for data in train_data:
            # print("========",111)
            loss, logits, layer_logits = model_engine(data)

            model_engine.backward(loss)
            model_engine.step()
            __tqdm.set_postfix(f"loss={round(loss,4)}")
            __tqdm.update(1)
