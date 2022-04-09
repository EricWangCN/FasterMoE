#!/bin/bash

D_MODEL=1024 \
TRACE_PATH=$DUMP_PREFIX/moe-gpt,$DUMP_PREFIX/moe-bert,$DUMP_PREFIX/gshard-gpt \
TRACE_LAYER=0,4,8 \
TRACE_ITER=500,10500,40500 \
MASTER_PORT=$(expr $RANDOM % 10000 + 10000) \
python3 -m torch.distributed.launch --nproc_per_node=1 /mnt/t-zilongwang/FasterMoE/benchmarks/breakdown_trace.py | tee logs/estm.log | grep Running 



