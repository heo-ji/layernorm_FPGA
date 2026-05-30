echo "############ custom_GELU , base2softmax , custom_invsqrt_norm -> sst2 (8.8) ############"
python3 run_glue.py \
  --model_name_or_path ModelTC/bert-base-uncased-sst2 \
  --task_name sst2  \
  --do_eval \
  --max_seq_length 128 \
  --output_dir ../../../results_bert_base/sst2/ \
  --overwrite_output_dir \
  --softmax_method base2  \
    --hidden_act CustomGELU \
    --layernorm_method custom_invsqrt_norm