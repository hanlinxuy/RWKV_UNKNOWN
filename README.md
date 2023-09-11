# RWKV_UNKNOWN

This is my implementation of RWKV language model.

```
cd src
deepspeed main.py --deepspeed --deepspeed_config=./configs/ds_config.config
```
# TODO
- [ ] jsonl data loading
- [ ] ckpt saving/load
- [ ] logging dump
- [ ] config dump
- [ ] megatron
- [ ] rwkv5
- [ ] attention mask


# Reference
- [RWKV](https://github.com/BlinkDL/RWKV-LM): Parallelizable RNN with Transformer-level LLM Performance (pronounced as "RwaKuv", from 4 major params: R W K V) 
- Data preprocessor from [TrainChatGalRWKV](https://github.com/SynthiaDL/TrainChatGalRWKV/)
- [neromous](https://github.com/neromous/) for the initaial code.
- [RWKV-infctx-trainer](https://github.com/RWKV/RWKV-infctx-trainer) for the model initialization code.
