#!/usr/bin/env bash
set -e

cd "$HOME/gitrepository/layernorm_FPGA/PyTorch_BERT_Base_trace"

for task in qnli qqp rte cola sst2 stsb mnli mrpc; do
    python3 tensor_2_hex.py \
        --tensor_dir "${task}/tensor" \
        --parallel_row_num 32 \
        --batch_size 8 \
        --device cuda \
        --num_forwards 1 5 10 20 30 40
done
