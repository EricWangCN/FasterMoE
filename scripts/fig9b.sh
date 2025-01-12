#!/bin/bash

echo >logs/pred.log

for tp in $DUMP_PREFIX/moe-gpt $DUMP_PREFIX/moe-bert $DUMP_PREFIX/gshard-gpt
do
    echo "Running trace $tp"
    for tl in 0 5 10
    do
        for ti in 500 40500
        do
            export D_MODEL=1024,2048,3072
            export TRACE_PATH=$tp
            export TRACE_LAYER=$tl
            export TRACE_ITER=$ti
            export MASTER_PORT=$(expr $RANDOM % 10000 + 10000)
            python3 -m torch.distributed.launch --nproc_per_node=1 /mnt/t-zilongwang/FasterMoE/benchmarks/run_trace.py >> logs/pred.log
        done
    done
done
