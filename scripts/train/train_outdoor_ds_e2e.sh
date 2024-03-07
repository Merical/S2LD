#!/bin/bash -l

#conda activate sheli
python train.py configs/data/megadepth_trainval_640.py configs/s2ld/outdoor/s2ld_kp_ds_dense.py \
--exp_name=s2ld-kpnet-e2e-tri-ds-bz=4-gpus=1-size=640 \
--gpus=1 \
--num_nodes=1 \
--accelerator=ddp \
--batch_size=2 \
--num_workers=4 \
--pin_memory=true \
--check_val_every_n_epoch=1 \
--log_every_n_steps=1 \
--flush_logs_every_n_steps=1 \
--limit_val_batches=1. \
--num_sanity_val_steps=10 \
--benchmark=True \
--max_epochs=30