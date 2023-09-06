RWKV_VOCAB=WORLD python train.py \
  --load_model /root/autodl-fs/RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth  \
  --proj_dir out1b5_test --wandb ""\
  --data_file /root/autodl-tmp/dollytest_text_document \
  --data_type binidx \
  --vocab_size 65536 --ctx_len 4096 --epoch_steps 100 --epoch_count 20 \
  --epoch_begin 0 --epoch_save 1 --micro_bsz 1 --n_layer 24 --n_embd 2048 \
  --pre_ffn 0 --head_qk 0 --lr_init 1e-4 --lr_final 1e-5 --warmup_steps 1000 \
  --beta1 0.9 --beta2 0.999 --adam_eps 1e-8 --accelerator gpu --devices 1 \
  --precision fp16 --grad_cp 1 --accumulate_grad_batches 1 --strategy deepspeed_stage_2_offload 