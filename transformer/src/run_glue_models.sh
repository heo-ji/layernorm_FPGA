#!/bin/bash

# Task,Validation 샘플 수,Batch Size = 16,Batch Size = 8
# CoLA,"1,043개",66 배치,131 배치
# MNLI (matched),"9,815개",614 배치,"1,227 배치"
# MRPC,408개,26 배치,51 배치
# QNLI,"5,463개",342 배치,683 배치
# QQP,"40,430개","2,527 배치","5,054 배치"
# RTE,277개,18 배치,35 배치
# SST-2,872개,55 배치,109 배치
# STS-B,"1,500개",94 배치,188 배치

# Define arrays of model paths and task names
model_paths=("ModelTC/bert-base-uncased-qnli" \
            "ModelTC/bert-base-uncased-qqp" \
            "ModelTC/bert-base-uncased-rte" \
            "ModelTC/bert-base-uncased-cola" \
            "ModelTC/bert-base-uncased-sst2" \
            "ModelTC/bert-base-uncased-stsb" \
            "ModelTC/bert-base-uncased-mnli" \
            "ModelTC/bert-base-uncased-mrpc")

task_names=("qnli" "qqp" "rte" "cola" "sst2" "stsb" "mnli" "mrpc")

# Ensure both arrays have the same length
length=${#model_paths[@]}

DIR="../../result_bert_base" # bert_base (layernorm_FPGA 바로 아래)
#TENSOR_DIR="/content/drive/MyDrive/bert_output/GLUEtask_tensor" # forward_fxp88 중간 텐서 저장 경로 (드라이브
TENSOR_DIR="../../PyTorch_BERT_Base_trace" # 텐서 저장 경로 (layernorm_FPGA 바로 아래)

# Loop through each model path and task name
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
#CUDA_VISIBLE_DEVICES=0
#nvidia-smi
for (( i=0; i<${length}; i++ ))
do
    python3 run_glue.py \
    --model_name_or_path ${model_paths[$i]} \
    --task_name ${task_names[$i]} \
    --do_eval \
    --output_dir $DIR/${task_names[$i]}/ \
    --overwrite_output_dir \
    --max_seq_length 128 \
    --softmax_method "base2"  \
    --hidden_act "CustomGELU" \
    --tensor_save_dir $TENSOR_DIR/${task_names[$i]}/tensor/ \
    --layernorm_method profiling_pass2 \
    --saif_pass2_layers "0,1,2,3,4,5,6,7,8,9,10,11" \
    --saif_pass2_k 10 \
    --max_eval_samples 200
done
#--per_device_eval_batch_size 16  batch=8 기준 K의 1/4 정도로 작게 나올 가능성이 높음, -> 일단 8로 해보고  , 실제 샘플 개수로 환산하면(K × batch_size) 둘 다 거의 같은 숫자
#--eval_accumulation_steps 5 colab에서 cpu ram가득차는것 방지
#--max_eval_samples 200 코드가 제대로 도는지, report가 잘 생기는지만 빠르게 확인하고 싶은 스모크 테스트 용도

# ──────────────────────────────────────────────────────────────────
#  옵션 목록 (for문 안의 값을 직접 바꿔서 사용)
# ──────────────────────────────────────────────────────────────────
#   --max_seq_length 128  -> 512bit, 128bit 만족을 위해
# ──────────────────────────────────────────────────────────────────
#   --softmax_method: 'base2' , 'original'
#   --hidden_act: 'gelu' or 'CustomGELU'"}
# ──────────────────────────────────────────────────────────────────
#   --layernorm_method 옵션
#   original            : torch 기본 LayerNorm (F.layer_norm)
#   custom_invsqrt_norm : Q8.8 fixed-point SW 골든 모델. tensor 저장(_save_fxp_tensor 호출)은
#                         custom_norm.py의 forward_fxp88 안에서 현재 주석처리되어 있음.
#                         layer{N}_{atten|ffn}_{input,mean,invsqrt,normalized}.pt 저장이 필요하면
#                         해당 4줄의 주석을 풀 것 (--tensor_save_dir 에 저장, 매번 덮어쓰기됨)
#   dualpath_norm       : significant/trivial dimension 나눠서 계산하는 실험용 방식
#   hw_mode1            : ZCU111 HW LayerNorm 결과를 그대로 다음 레이어로 전달 (--hw_ip/--hw_port 필요)
#   hw_mode2            : SW golden으로 흐르되 HW 결과와 비교해서 worst-case 오차 입력 기록
#   profiling_pass1     : SAIF 대표 tensor 선정용 profiling. .pt tensor 저장은 없고, 위치별
#                         (layer x atten/ffn) activity(toggle rate+static prob)만 기록하다가
#                         eval 끝나면 --tensor_save_dir 에 {task_name}_convergence.json 저장
#                         (forward 수를 늘려가며 activity가 수렴하는 지점 K 확인용)
#   profiling_pass2     : --saif_pass2_layers (예: "0,1,5,9,11") 에 지정한 layer들의 atten+ffn에서
#                         forward #0~(--saif_pass2_k - 1)를 이어붙여 layer{N}_{atten|ffn}_{input,mean,invsqrt,normalized}.pt
#                         로 저장 (--tensor_save_dir 기준). 정확도 보는거 아니고, forward K개만 필요하므로 eval set도 K*batch_size개로 앞에서부터 딱 그만큼만 잘라낸 작은 데이터셋을 만들어서 그걸 trainer.evaluate()에 넘깁잘라서 돎
