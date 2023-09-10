# RWKV_UNKNOWN

This is my implementation of RWKV language model.


```
cd src
deepspeed main.py --deepspeed --deepspeed_config=./configs/ds_config.config
```

# RWKV: Parallelizable RNN with Transformer-level LLM Performance (pronounced as "RwaKuv", from 4 major params: R W K V)

Please visit: https://github.com/BlinkDL/RWKV-LM

# Reference
- For easy usage, I just copy data preprocessor from [TrainChatGalRWKV](https://github.com/SynthiaDL/TrainChatGalRWKV/)
- Thanks to [neromous](https://github.com/neromous/) for the initaial code.
- Thanks to [RWKV-infctx-trainer](https://github.com/RWKV/RWKV-infctx-trainer) for the model initialization code.