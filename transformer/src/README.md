# BERT GLUE 실행 / SAIF profiling

## 실행

- 위치: `src/`
- 대상: GLUE 8개 task 순차 실행

```bash
bash run_glue_models.sh
```

현재 옵션:

```bash
    --layernorm_method profiling_pass2 \
    --saif_pass2_layers "0,1,2,3,4,5,6,7,8,9,10,11" \
    --saif_pass2_representative_json "./saif_representative_forwards.json"
    # --saif_pass2_k 10
```

- 앞 K개 저장: 마지막 JSON 옵션 줄을 `--saif_pass2_k 10`으로 교체
- 공통: batch 8 / sequence 128 / 고정 padding / base2 softmax / CustomGELU
- 결과: `../../result_bert_base/{task}/`
- tensor: `../../PyTorch_BERT_Base_trace/{task}/tensor/`

## 주요 옵션

| 옵션 | 코드 기본값 | 용도 |
|---|---|---|
| `--model_name_or_path` | 필수 | 모델 경로 / Hugging Face 모델 ID |
| `--task_name` | `None` | `cola, mnli, mrpc, qnli, qqp, rte, sst2, stsb, wnli` |
| `--do_eval` | `False` | 평가 실행 |
| `--output_dir` | 필수 | 모델 / 평가 결과 경로 |
| `--overwrite_output_dir` | `False` | 기존 output 덮어쓰기 허용 |
| `--max_seq_length` | `128` | 최대 시퀀스 길이 |
| `--softmax_method` | `original` | `base2` / `original` / `cordic`(미사용) |
| `--hidden_act` | `gelu` | `gelu` / `CustomGELU` |
| `--layernorm_method` | `original` | 아래 LayerNorm 방식 참고 |
| `--tensor_save_dir` | `GLUEtask_tensor` | pass1 JSON / pass2 tensor 경로 |
| `--saif_pass2_layers` | `0,1,5,9,11` | 저장할 layer 목록; 스크립트: 0~11 전체 |
| `--saif_pass2_k` | `40` | 처음 K개 forward 누적 저장; 스크립트 주석 예시: 10 |
| `--saif_pass2_representative_json` | `None` | 위치별 대표 forward 인덱스 JSON; 지정 시 K 옵션 무시 |

## LayerNorm 방식

| `--layernorm_method` | 용도 |
|---|---|
| `original` | PyTorch 기본 LayerNorm |
| `custom_invsqrt_norm` | Q8.8 fixed-point SW golden (`forward_fxp88`) |
| `dualpath_norm` | significant / trivial dimension 분리 실험 |
| `hw_mode1` | HW 출력의 다음 layer 전달 / HW accuracy 확인 |
| `hw_mode2` | SW golden 유지 / HW 오차 비교 / worst-case 입력 기록 |
| `profiling_pass1` | 전체 validation 입력 activity 분석 / 수렴 K·대표 인덱스 선정 |
| `profiling_pass2` | ① 처음 K개 forward 누적 저장 ② 위치별 대표 인덱스 k의 배치 하나 저장 |

## 코드 변경기록

| 파일 | 구분 | 핵심 변경 |
|---|---|---|
| `custom_norm.py` | pass1 이전 | `forward_fxp88`에서 input/mean/invsqrt/normalized 저장; 같은 파일 덮어쓰기; 현재 저장 호출 주석 처리 |
| `custom_norm.py` | pass1 | 입력 32차원 activity / 평균 / 수렴 K / 대표 forward 인덱스 기록 |
| `custom_norm.py` | pass2: 누적 K개 | forward #0~K-1 tensor를 batch 차원으로 누적·연결 저장 |
| `custom_norm.py` | pass2: 대표 위치 k | JSON의 위치별 forward 인덱스 k에서 배치 하나 저장 |
| `representative_json_file_checker.py` | JSON 검증 | task 선택 / 누락 위치 / 인덱스 범위 / 기록된 실행 설정 검사 |
| `run_glue_models.sh` | 실행 옵션 | 대표 JSON 옵션 활성화 / 기존 K 옵션 주석 유지 |
| `custom_invsqrt.py` | SW/RTL LUT 일치 | raw CSV + floor quantization → `_fixed.csv` hex parsing |
| `custom_invsqrt.py` | LUT index 오류 방지 | `pd.read_csv(..., header=None)` |

## SAIF profiling

- 구현: [custom_norm.py](transformers/models/bert/custom_norm.py)
- 위치: layer 0~11 × atten/ffn = 24곳
- forward: 배치 1개 / **K: 개수 / k: 0-based 인덱스**
- 스크립트의 `--layernorm_method`로 pass1 / pass2 선택

### pass1: 입력 activity 분석

- 범위: 전체 validation; `--max_eval_samples` 지정 시 제한된 범위
- 입력: Q8.8 양자화
- 통계: feature 방향 비트별 toggle rate 16개 + static probability 16개 = 32차원
- 출력: `{tensor_save_dir}/{task}_convergence.json`; `.pt` 저장 없음

| JSON 항목 | 의미 |
|---|---|
| `num_forwards` | 전체 배치 수 N |
| `mean_activity` | 전체 forward activity 평균 |
| `medoid_forward_idx` | 전체 평균에 가장 가까운 실제 forward 인덱스 |
| `medoid_distance` | 선택된 forward와 전체 평균의 RMS 거리 |
| `convergence_forward_count` | 누적 평균이 최종 평균과 epsilon 이내로 유지되는 최소 K |

- epsilon: 현재 코드 기본값 `0.005`; 실제 기준은 report의 `epsilon` 확인
- 수렴 K: 입력 activity 평균 기준; 전력 수렴 보장 없음

### pass2: tensor 저장

**두 가지 지원:**

| 방식 | 옵션 | 저장 대상 |
|---|---|---|
| 1. 누적 K개 forward | JSON 미지정 + `--saif_pass2_k K` | 처음 #0~K-1 배치를 연결한 tensor |
| 2. 대표 위치 k | `--saif_pass2_representative_json 파일` | 각 layer/atten/ffn의 지정 인덱스 k 배치 하나 |

- 누적 K개: 앞 `K × batch_size`개 샘플 평가 / 데이터 부족 시 모인 만큼 저장
- 대표 위치 k: 최대 선택 인덱스까지 순차 평가 / 지정 배치만 즉시 저장 / K 옵션 무시
- 대표 JSON: `saif_representative_forwards.json` — 8개 task × 24곳; `--task_name`으로 선택
- 기존 pass1 `{task}_convergence.json` 직접 입력도 지원
- 예: QNLI layer0 atten → #294 / ffn → #532
- 대표 모드 조건: pass1과 동일 모델·데이터 순서·전처리 / batch 8 / sequence 128 / 고정 padding / 단일 디바이스
- 대표 모드: `--max_eval_samples` 제한 없이 실행
- JSON 검사: `representative_json_file_checker.py`; 대표 인덱스 재선정 없음

### 공통 저장 형식

```text
{tensor_save_dir}/layer{N}_{atten|ffn}_{input|mean|invsqrt|normalized}.pt
```

- 전체 24곳 × 4종 = task당 96개 파일
- 두 방식 모두 같은 파일명; 같은 폴더 재실행 시 덮어쓰기

| tensor | 대표 배치 하나 | 누적 K개 배치 |
|---|---|---|
| input / normalized | `(8,128,768)` | `(K*8,128,768)` |
| mean / invsqrt | `(8,128,1)` | `(K*8,128,1)` |

- 마지막 불완전 배치: 실제 샘플 수에 따른 첫 차원 축소
- hex 변환: 대표 모드 `--batch_size 8 --num_forwards 1` / 누적 모드 저장된 K 이내 선택

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

필수 입력: `task_name` / `dataset_name` / `train_file`+`validation_file` 중 하나

### TrainingArguments (HuggingFace 표준, 자주 쓰이는 것만 참고용으로)

전체 옵션: `python run_glue.py --help` / `transformers/training_args.py`

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