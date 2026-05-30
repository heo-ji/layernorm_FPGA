#!/bin/bash

# # Define arrays of model paths and task names
# model_paths=("ModelTC/bert-base-uncased-mnli" \
#             "ModelTC/bert-base-uncased-mrpc" \
#             "ModelTC/bert-base-uncased-qnli" \
#             "ModelTC/bert-base-uncased-qqp" \
#             "ModelTC/bert-base-uncased-rte" \
#             "ModelTC/bert-base-uncased-sst2" \
#             "ModelTC/bert-base-uncased-stsb" \
#             "ModelTC/bert-base-uncased-cola")

# task_names=("mnli" "mrpc" "qnli" "qqp" "rte" "sst2" "stsb" "cola" )

# # Ensure both arrays have the same length
# length=${#model_paths[@]}

# DIR="/home/user/HJH/results_bert_base/" # bert_base

# # Loop through each model path and task name
# for (( i=0; i<${length}; i++ ))
# do
#     python3 run_glue.py \
#     --model_name_or_path ${model_paths[$i]} \
#     --task_name ${task_names[$i]} \
#     --do_eval \
#     --max_seq_length 128 \  
#       --max_eval_samples 200 \
#     --output_dir $DIR/${task_names[$i]}/ \
#     --overwrite_output_dir \
#     --softmax_method base2  \
#     --hidden_act "CustomGELU"
# done




echo "############ original -> mrpc (8.8)############"
python3 run_glue.py \
  --model_name_or_path ModelTC/bert-base-uncased-mrpc \
  --task_name mrpc  \
  --do_eval \
  --max_seq_length 128 \
  --output_dir /home/user/HJH/results_bert_base/mrpc/ \
  --overwrite_output_dir \
  --softmax_method original  \
    --hidden_act gelu \
    --layernorm_method original

echo "############ custom_GELU , base2softmax , custom_invsqrt_norm -> mrpc (8.8) ############"
python3 run_glue.py \
  --model_name_or_path ModelTC/bert-base-uncased-mrpc \
  --task_name mrpc  \
  --do_eval \
  --max_seq_length 128 \
  --output_dir /home/user/HJH/results_bert_base/mrpc/ \
  --overwrite_output_dir \
  --softmax_method base2  \
    --hidden_act CustomGELU \
    --layernorm_method custom_invsqrt_norm

# # echo "############ dual path norm -> mrpc ############"
# python3 run_glue.py \
#   --model_name_or_path ModelTC/bert-base-uncased-mrpc \
#   --task_name mrpc  \
#   --do_eval \
#   --max_seq_length 128 \
#   --output_dir /home/user/HJH/results_bert_base/mrpc/ \
#   --overwrite_output_dir \
#   --softmax_method original  \
#     --hidden_act gelu \
#     --layernorm_method dualpath_norm

