# run_glue.py 실행 인자 정리


## 실제 사용 중인 옵션 (./run_glue_models.sh 기준)

```bash
DIR="../../result_bert_base"          # layernorm_FPGA 바로 아래에 생성
TENSOR_DIR="../../GLUEtask_tensor"    # layernorm_FPGA 바로 아래에 생성

python3 run_glue.py \
    --model_name_or_path ${model_paths[$i]} \
    --task_name ${task_names[$i]} \
    --do_eval \
    --max_seq_length 128 \
    --output_dir $DIR/${task_names[$i]}/ \
    --overwrite_output_dir \
    --softmax_method "base2"  \
    --hidden_act "CustomGELU" \
    --layernorm_method custom_invsqrt_norm \
    --tensor_save_dir $TENSOR_DIR/${task_names[$i]}/
```

| 옵션 | 기본값 | 소속 | 설명 |
|---|---|---|---|
| `--model_name_or_path` | (필수) | ModelArguments | 사전학습 모델 경로 또는 huggingface.co/models 식별자 |
| `--task_name` | `None` | DataTrainingArguments | GLUE task 이름: `cola, mnli, mrpc, qnli, qqp, rte, sst2, stsb, wnli` 중 하나 |
| `--do_eval` | `False` | TrainingArguments | 평가 실행 여부 |
| `--max_seq_length` | `128` | DataTrainingArguments | 토큰화 후 최대 시퀀스 길이 (넘으면 truncate, 짧으면 pad) |
| `--output_dir` | (필수) | TrainingArguments | 체크포인트/결과 저장 경로 |
| `--overwrite_output_dir` | `False` | TrainingArguments | `output_dir`가 비어있지 않아도 덮어쓸지 여부 |
| `--softmax_method` | `"original"` | ModelArguments | softmax 방식: `cordic`(사용 x) / `base2` / `original` |
| `--hidden_act` | `"gelu"` | ModelArguments | GELU 방식: `gelu` 또는 `CustomGELU` |
| `--layernorm_method` | `"original"` | ModelArguments | LayerNorm 방식: `original` / `custom_invsqrt_norm` / `dualpath_norm` / `hw_mode1` / `hw_mode2` |
| `--tensor_save_dir` | `"GLUEtask_tensor"` | ModelArguments | forward_fxp88 중간 텐서 저장 경로 (아래 섹션 참고) |

---


# forward_fxp88 중간 텐서 저장 (`GLUEtask_tensor/`)

`layernorm_method custom_invsqrt_norm`(→ `Custom_LayerNorm.forward_fxp88`,
[transformers/models/bert/custom_norm.py](transformers/models/bert/custom_norm.py))으로 평가할 때,
아래 4개 중간 텐서 (Q8.8)를 레이어/블록별로 파일에 저장한다.
- `input_fx16`
- `mean`
- `invsqrt`
- `normalized`

```
eval 스텝(=DataLoader가 몇 번 forward를 호출했는지)
GLUE 평가셋 전체(수백~수천 개)를 미니배치 8개씩(텐서안의 B) 나눠서 forward_fxp88을 여러 번 호출함. 마지막 스텝에서의 미니배치 8개만 파일에 남고 그 이전 스텝의 값은 덮어써져서 사라짐.
실제로 전체 eval셋을 다 남기고 싶다면 _save_fxp_tensor에 스텝 카운터를 넣어 layer0_atten_input_step3.pt처럼 파일명에 스텝 번호를 붙이거나, 매 스텝 텐서를 리스트에 모았다가 torch.cat(dim=0) 해서 eval 끝에 한 번만 저장하는 방식으로 바꿔야함
```

## 저장 경로

```
{tensor_save_dir}/layer{N}_{block_type}_{tensor_name}.pt
```

- `{tensor_save_dir}`: `run_glue.py`에 넘긴 `--tensor_save_dir` 값. 지정하지 않으면 실행 위치 기준 상대경로 `GLUEtask_tensor`.
- `{N}`: BERT layer 번호 (0-base, `config.num_hidden_layers-1`까지).
- `{block_type}`: `atten`(self-attention 뒤 LayerNorm, `BertSelfOutput`) / `ffn`(FFN 뒤 LayerNorm, `BertOutput`) / `crossatten`(decoder cross-attention, 이 repo에선 사용 안 함).
- `{tensor_name}`: `input` / `mean` / `invsqrt` / `normalized`.

`run_glue_models.sh`는 `--tensor_save_dir $TENSOR_DIR/${task_names[$i]}/`로 task별 하위 폴더까지 직접 지정해서 넘긴다.
`TENSOR_DIR="../../GLUEtask_tensor"`이므로 `src/`에서 스크립트를 실행하면 최종 경로는 `layernorm_FPGA` 바로 아래에 생긴다:

```
GLUEtask_tensor/mrpc/layer0_atten_input.pt
GLUEtask_tensor/mrpc/layer0_atten_mean.pt
GLUEtask_tensor/mrpc/layer0_atten_invsqrt.pt
GLUEtask_tensor/mrpc/layer0_atten_normalized.pt
GLUEtask_tensor/mrpc/layer0_ffn_input.pt
...
GLUEtask_tensor/mrpc/layer11_ffn_normalized.pt
```



## torch.save를 끄고 싶을 때
`custom_norm.py`의 `Custom_LayerNorm.forward_fxp88` 안에서 아래 4줄을 찾아 주석 처리:

```python
self._save_fxp_tensor(input_fx16, 'input')
...
self._save_fxp_tensor(mean, 'mean')
...
self._save_fxp_tensor(invsqrt, 'invsqrt')
...
self._save_fxp_tensor(normalized, 'normalized')
```













## 참고용 옵션 (run_glue_models.sh에서는 안 씀)
### ModelArguments

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--config_name` | `None` | model_name과 다른 config를 쓰고 싶을 때 경로/이름 |
| `--tokenizer_name` | `None` | model_name과 다른 tokenizer를 쓰고 싶을 때 경로/이름 |
| `--cache_dir` | `None` | huggingface.co에서 받은 사전학습 모델을 저장할 캐시 경로 |
| `--use_fast_tokenizer` | `True` | tokenizers 라이브러리 기반 fast tokenizer 사용 여부 |
| `--model_revision` | `"main"` | 사용할 모델 버전(브랜치명/태그명/커밋id) |
| `--token` | `None` | private repo 접근용 HTTP bearer 토큰 (없으면 `huggingface-cli login` 토큰 사용) |
| `--use_auth_token` | `None` | (deprecated) `--token` 사용 권장 |
| `--trust_remote_code` | `False` | Hub에 있는 커스텀 모델링 코드를 실행할지 여부 (신뢰하는 repo에만 True) |
| `--ignore_mismatched_sizes` | `False` | 헤드 차원이 다른 사전학습 모델도 로드 허용 |
| `--hw_ip` | `"166.104.140.13"` | ZCU111 PS IP 주소 (`hw_mode1`/`hw_mode2`에서만 사용) |
| `--hw_port` | `5000` | ZCU111 PS TCP 포트 (`hw_mode1`/`hw_mode2`에서만 사용) |

### DataTrainingArguments

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--dataset_name` | `None` | (task_name 대신) datasets 라이브러리에서 쓸 데이터셋 이름 |
| `--dataset_config_name` | `None` | 위 dataset의 config 이름 |
| `--overwrite_cache` | `False` | 전처리된 데이터셋 캐시를 덮어쓸지 여부 |
| `--pad_to_max_length` | `True` | 모든 샘플을 `max_seq_length`로 패딩할지, 배치 단위 동적 패딩할지 |
| `--max_train_samples` | `None` | 디버깅/속도용으로 학습 샘플 수 제한 |
| `--max_eval_samples` | `None` | 디버깅/속도용으로 평가 샘플 수 제한 |
| `--max_predict_samples` | `None` | 디버깅/속도용으로 예측 샘플 수 제한 |
| `--train_file` | `None` | 커스텀 학습 데이터 csv/json 경로 |
| `--validation_file` | `None` | 커스텀 검증 데이터 csv/json 경로 |
| `--test_file` | `None` | 커스텀 테스트 데이터 csv/json 경로 |

`task_name`, `dataset_name`, (`train_file`+`validation_file`) 중 하나는 반드시 지정해야 한다.

### TrainingArguments (HuggingFace 표준, 자주 쓰이는 것만 참고용으로)

전체 목록은 `python run_glue.py --help` 또는 `transformers/training_args.py` 참고.

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--do_train` | `False` | 학습 실행 여부 |
| `--do_predict` | `False` | 예측(테스트셋) 실행 여부 |
| `--per_device_train_batch_size` | `8` | GPU/디바이스당 학습 배치 크기 |
| `--per_device_eval_batch_size` | `8` | GPU/디바이스당 평가 배치 크기 |
| `--learning_rate` | `5e-5` | 학습률 |
| `--num_train_epochs` | `3.0` | 학습 epoch 수 |
| `--seed` | `42` | 랜덤 시드 |

---