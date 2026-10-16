# run_glue.py 실행 인자 정리


## 실제 사용 중인 옵션 (./run_glue_models.sh 기준)

```bash
# src/에서 실행: 대표 JSON의 위치별 배치 하나 저장 (8개 GLUE task 순차 실행)
bash run_glue_models.sh
```

스크립트의 마지막 활성 옵션은 아래와 같다. 기존 앞 K개 옵션은 바로 아래에 주석으로 남겨 두었다.

```bash
    --saif_pass2_representative_json "./saif_representative_forwards.json"
    # --saif_pass2_k 10
```

앞 K개 저장으로 바꾸려면 **마지막 JSON 옵션 줄을 `--saif_pass2_k 10`으로 교체**한다.
JSON 파일 경로를 바꾸려면 같은 줄의 경로를 직접 수정한다.


공통 설정은 batch size 8, max sequence length 128, 고정 길이 padding,
`softmax_method=base2`, `hidden_act=CustomGELU`, `layernorm_method=profiling_pass2`,
layer 0~11의 atten/ffn 전체이다. `src/`에서 실행하며 상대경로는 실행 위치를 기준으로 한다.
`DIR="../../result_bert_base"`, `TENSOR_DIR="../../PyTorch_BERT_Base_trace"`이며,
저장 경로는 `$TENSOR_DIR/{task}/tensor/`이다. 두 모드의 차이는 아래 pass2 설명 참고.


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
| `--layernorm_method` | `"original"` | ModelArguments | LayerNorm 방식: `original` / `custom_invsqrt_norm` / `dualpath_norm` / `hw_mode1` / `hw_mode2` / `profiling_pass1` / `profiling_pass2` |
| `--tensor_save_dir` | `"GLUEtask_tensor"` | ModelArguments | forward_fxp88 중간 텐서 저장 경로. `profiling_pass1`의 수렴(K) report, `profiling_pass2`의 대표 tensor도 이 경로에 저장됨 (아래 섹션 참고) |
| `--saif_pass2_layers` | `"0,1,5,9,11"` | ModelArguments | `profiling_pass2`에서 tensor를 저장할 layer_idx 목록 (comma-separated) |
| `--saif_pass2_k` | `40` | ModelArguments | JSON 미지정 시 처음 K개 forward를 이어붙여 저장. 스크립트에는 기존 `--saif_pass2_k 10` 옵션을 주석으로 남김 |
| `--saif_pass2_representative_json` | `None` | ModelArguments | 위치별 대표 forward 인덱스 JSON 경로. 지정하면 `--saif_pass2_k`를 무시하고 위치마다 한 배치 저장 |

---


# forward_fxp88 중간 텐서 저장 (`GLUEtask_tensor/`)

`layernorm_method custom_invsqrt_norm`(→ `Custom_LayerNorm.forward_fxp88`,
[transformers/models/bert/custom_norm.py](transformers/models/bert/custom_norm.py))으로 평가할 때,
아래 4개 중간 텐서 (Q8.8)를 저장할 수 있다. 현재 `forward_fxp88`의 저장 호출은 주석 처리되어 있으며,
주석을 해제하면 아래처럼 마지막 배치가 남는다. `profiling_pass2`의 저장 기능은 별도로 동작한다.
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

`run_glue_models.sh`는 `--tensor_save_dir "$TENSOR_DIR/${task_names[$i]}/tensor/"`를 넘긴다.
`TENSOR_DIR="../../PyTorch_BERT_Base_trace"`이므로 `layernorm_FPGA` 바로 아래에 생성된다.

```text
PyTorch_BERT_Base_trace/mrpc/tensor/layer0_atten_input.pt
PyTorch_BERT_Base_trace/mrpc/tensor/layer0_atten_mean.pt
PyTorch_BERT_Base_trace/mrpc/tensor/layer0_atten_invsqrt.pt
PyTorch_BERT_Base_trace/mrpc/tensor/layer0_atten_normalized.pt
...
PyTorch_BERT_Base_trace/mrpc/tensor/layer11_ffn_normalized.pt
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

# 코드 변경기록
| 파일                  | 변경 목적                   | 핵심 변경                                                            |
| ------------------- | ----------------------- | ---------------------------------------------------------------- |
| `custom_norm.py`    | SAIF 입력 대표성 확보위해 pass2:텐서저장          | activity 기반 representative forward만 tensor 저장 |
| `custom_norm.py`    | SAIF 입력 대표성 확보위해 프로파일링pass1          |입력 switching 수렴 forward 수 K  |
| `custom_invsqrt.py` | SW/RTL LUT bit-exact 일치 | raw CSV + floor quantization → `_fixed.csv` hex 직접 parsing       |
| `custom_invsqrt.py` | LUT index 오류 방지         | `pd.read_csv(..., header=None)` 적용                               |












# SAIF profiling (`profiling_pass1` / `profiling_pass2`)

BERT 각 LayerNorm 위치(layer × atten/ffn, 24곳)의 입력 activity를 pass1에서 분석하고,
pass2에서 대표 배치 또는 처음 K개 배치의 실제 tensor를 SAIF 입력용으로 저장한다.
[transformers/models/bert/custom_norm.py](transformers/models/bert/custom_norm.py)에 구현.

`run_glue_models.sh`의 for문 안 `--layernorm_method` 값을 `profiling_pass1`로 직접 바꿔서 실행
(옵션 목록은 스크립트 맨 아래 주석 참고).

## pass1 (`--layernorm_method profiling_pass1`, `Custom_LayerNorm.forward_profiling_pass1`)

- 위치별로 LayerNorm 입력을 Q8.8로 변환 후, feature(col) 방향 인접 bit toggle rate(16개) +
  static probability(16개) = 32차원 activity vector를 forward마다 기록만 함 (`.pt` tensor 저장 없음).
- `trainer.evaluate()` 종료 후 `Custom_LayerNorm.saif_write_pass1_report()`가
  forward 1, 2, 4, 8, 16, 32, ... 개 누적했을 때의 running-mean activity를
  전체 평균과 비교해서, 그 이후로 계속 RMS 오차(epsilon, 현재 코드 기본값 0.005)
  이내에 머무는 가장 작은 forward 개수
  (`convergence_forward_count`, K)를 위치별로 계산.
  참고용으로 전체 평균에 가장 가까운 실제 forward(`medoid_forward_idx`)와 전체 평균
  activity vector(`mean_activity`, 32차원)도 같이 기록. `mean_activity`는 위치 간(예: layer끼리)
  activity가 실제로 비슷한지 비교하는 데 씀 - K는 위치 내부 수렴 속도일 뿐 위치 간 유사도를 말해주지 않음.
- 결과는 `--tensor_save_dir`로 넘긴 경로에 `{task_name}_convergence.json`으로 저장
  (task별로 파일 분리되므로 `run_glue_models.sh`로 여러 task를 루프 돌리면 task별 report가 각각 생김).

```
GLUEtask_tensor/mnli/mnli_convergence.json
GLUEtask_tensor/sst2/sst2_convergence.json
...
```

## pass2 (`--layernorm_method profiling_pass2`, `Custom_LayerNorm.forward_profiling_pass2`)

**두 모드 모두 지원한다.** forward는 샘플 하나가 아니라 배치 하나다.
`--saif_pass2_layers`로 저장할 layer를 선택하며 각 layer의 atten/ffn 모두 저장한다.
기본 layer 목록은 `0,1,5,9,11`이고, `run_glue_models.sh`는 0~11 전체를 지정한다.

### 대표 JSON 모드 (스크립트 기본값)

- `--saif_pass2_representative_json`으로 JSON을 지정하면 위치별 지정 배치 하나를 즉시 저장한다.
  JSON 숫자는 **0-based forward 인덱스**다. 예: QNLI `layer0_atten=294`, `layer0_ffn=532`.
- 제공된 `saif_representative_forwards.json`에는 pass1 보고서의 `medoid_forward_idx`를 옮긴
  8개 task × 24개 위치가 있다. `--task_name`으로 task를 선택한다.
  기존 `{task}_convergence.json`도 직접 지정할 수 있다.
- `representative_json_file_checker.py`는 JSON 읽기와 누락 위치·인덱스·기록된 실행 설정 검증을 담당한다.
  대표 forward를 새로 선정하지 않는다. `run_glue.py`와 같은 폴더에 둔다.
- 이 모드에서는 `--saif_pass2_k`를 무시한다. 원래 인덱스를 유지하도록 처음부터 선택된 최대 인덱스까지
  평가하되, 선택하지 않은 배치 tensor는 저장하거나 누적하지 않는다. `--max_eval_samples`로 제한하지 않는다.

### 앞 K개 모드

- JSON 옵션을 생략하고 `--saif_pass2_k K`를 지정하면 forward `#0~K-1`을 batch 차원으로 이어붙여 저장한다.
- `run_glue.py`의 기본 K는 40이다. 스크립트에서는 마지막 JSON 옵션 줄을 `--saif_pass2_k 10`으로 교체하여 실행한다.
- 평가 데이터는 앞 `K * per_device_eval_batch_size`개만 사용한다. 데이터가 부족하면 모인 만큼 저장한다.
- pass1의 `convergence_forward_count`는 입력 activity 평균의 수렴 기준이며, 전력 수렴을 보장하는 값은 아니다.

### 공통 저장 형식과 실행 조건

- 파일명은 두 모드 모두 `layer{N}_{atten|ffn}_{input|mean|invsqrt|normalized}.pt`이다.
  같은 저장 폴더에서 다시 실행하면 같은 이름의 파일을 덮어쓴다. 전체 24곳이면 task당 96개 파일이다.
- batch size 8, 길이 128 기준 대표 모드의 input/normalized는 `(8,128,768)`, mean/invsqrt는 `(8,128,1)`이다.
  앞 K개 모드는 첫 차원이 `K*8`이며, 마지막 불완전 배치가 포함되면 실제 샘플 수에 맞춰 작아진다.
- 대표 JSON의 실행 조건은 batch size 8, max sequence length 128, 고정 길이 padding이다.
  pass1과 모델·데이터 순서·전처리·양자화 설정을 동일하게 하고 단일 프로세스/디바이스에서 실행한다.
  원본 pass1 보고서에는 배치/토큰화 설정이 없으므로 해당 설정은 직접 맞춰야 한다.
- 대표 모드 출력의 hex 변환은 `--batch_size 8 --num_forwards 1`을 사용한다.
  앞 K개 모드는 저장한 K 범위 내에서 `--num_forwards 1 2 5 10`처럼 변환한다.

---

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