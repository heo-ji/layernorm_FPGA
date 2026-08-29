#!/bin/bash

# Define arrays of model paths and task names
model_paths=("ModelTC/bert-base-uncased-mnli" \
            "ModelTC/bert-base-uncased-qnli" \
            "ModelTC/bert-base-uncased-qqp" \
            "ModelTC/bert-base-uncased-rte" \
            "ModelTC/bert-base-uncased-sst2" \
            "ModelTC/bert-base-uncased-stsb" \
            "ModelTC/bert-base-uncased-cola" \
            "ModelTC/bert-base-uncased-mrpc")

task_names=("mnli" "qnli" "qqp" "rte" "sst2" "stsb" "cola" "mrpc")

# Ensure both arrays have the same length
length=${#model_paths[@]}

DIR="../../result_bert_base" # bert_base (layernorm_FPGA 바로 아래)
TENSOR_DIR="../../GLUEtask_tensor" # forward_fxp88 중간 텐서 저장 경로 (layernorm_FPGA 바로 아래)

# 사용법:
#   ./run_glue_models.sh              → 기존과 동일 (custom_invsqrt_norm, tensor 저장)
#   ./run_glue_models.sh profiling    → SAIF profiling_pass1 (task별 수렴 report만 생성, tensor 저장 없음)
MODE=${1:-normal}
if [ "$MODE" == "profiling" ]; then
    LAYERNORM_METHOD="profiling_pass1"
else
    LAYERNORM_METHOD="custom_invsqrt_norm"
fi

# Loop through each model path and task name
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
for (( i=0; i<${length}; i++ ))
do
    CUDA_VISIBLE_DEVICES=0
    python3 run_glue.py \
    --model_name_or_path ${model_paths[$i]} \
    --task_name ${task_names[$i]} \
    --do_eval \
    --max_seq_length 128 \
    --output_dir $DIR/${task_names[$i]}/ \
    --overwrite_output_dir \
    --softmax_method "base2"  \
    --hidden_act "CustomGELU" \
    --layernorm_method $LAYERNORM_METHOD \
    --tensor_save_dir $TENSOR_DIR/${task_names[$i]}/
done



# python run_glue.py \
#   --model_name_or_path ModelTC/bert-base-uncased-mrpc \
#   --task_name mrpc  \
#   --do_eval \
#   --max_seq_length 128 \
#   --max_eval_samples 200 \
#   --output_dir /home/user/HJH/results_bert_base/mrpc/ \
#   --overwrite_output_dir \
#   --softmax_method original  \
#     --hidden_act gelu
# --max_seq_length 12800


#     softmax_method: 'base2' , 'original'
#     hidden_act: 'gelu' or 'CustomGELU'"}
