# layernorm_FPGA
fpga환경에서 HIL 환경 구축
# LayerNorm HIL (Hardware-in-the-Loop) 실행

[ Project : End-to-end transformer encoder FPGA accelerator ]
1. **SW** : [`repository [e2e-bert-accel-SW]`](https://github.com/heo-ji/e2e-bert-accel-SW)
2. **layernorm HW** : [`repository [layernorm_HW]`](https://github.com/heo-ji/layernorm_HW)
    - IP 코어 (데이터 포맷, 비트폭, 파라미터) 설명 [`링크`](https://github.com/heo-ji/layernorm_HW/blob/main/doc/README_HW.md)
    - AXI wrapper (레지스터 맵, FSM, 제어/데이터전송 시퀀스) 설명 [`링크`](https://github.com/heo-ji/layernorm_HW/blob/main/doc/README_AXI_WRAPPER.md)
    - HIL 환경 overview (Host PC ↔ ZCU111 board ) 설명 [`링크`](https://github.com/heo-ji/layernorm_HW/blob/main/doc/README_FPGA_overview.md)

---

# 사전 준비
1. SW BERT 모델
```
[e2e-bert-accel-SW] repository의 "transformers"에서 custom_norm.py, run_glue.py수정 한것임


layernorm_FPGA\transformer\
├── setup.py
├── pyproject.toml
└── src\
    ├── transformers\       
        ├── models\bert
            ├── modeling_bert_custom.py         
            │   ├── custom_norm.py               
            │   ├── custom_invsqrt.py              
            │   ├── custom_softmax_fixed.py         
            │   ├── custom_activations.py           
            │   ├── custom_configuration_bert.py    
            │   └── invsqrt_csv_file/               ← invsqrt LUT 테이블 
        │
        ├── ...등등

    ├── run_glue.py
    ├── run_glue_FXP_sst2.sh
```

```
conda create -n bert_hw python=3.10
conda activate bert_hw

cd transformer
pip install -e . //global python에서 하지 않도록 주의!
cd src
bash run_glue_FXP_sst2.sh
```
2. Bitstream, PS 제어코드 준비

```
layernorm_FPGA\
├── zcu111_ps_pynq.py          ← PS의 PL제어
├── FPGA_block_design/
│   ├── layernorm_HIL.bit
│   └── layernorm_HIL.hwh
└── transformer/src/
    ├── run_glue_FXP_sst2.sh    ← sst2 실행용 bash파일
    ├── run_glue.py             ← --hw_ip, --hw_port 인수 추가됨 , --
    └── transformers/models/bert/
        └── custom_norm.py      ← TCP client(HWLayerNormClient) & --layernorm_method로 "hw_mode1/2" 추가됨

```
zcu111보드의 pynq버전 2.1이상이면
PYNQ에서는 .bit + .hwh 두 개 (파일명 base name이 같게!) 옮겨놓는다.


# 실행 방법
### [1] 보드 부팅 ,ssh 연결
1. SD카드 : PYNQ 이미지카드 삽입
2. 보드를 스위치 booting mode (1110)로 셋팅 + SD카드 삽입 해서 노트북 usb연결시켜
3.  service ssh start 하면 166.104.140.13 서버에 ssh 접속 가능해짐
```
참고로
DNS가 고정이 잘 안되면 /etc/vi ..뭐 확인해서 수정하라는데 뭔지 모르겠음

# 노트북에서 (장치 이름은 환경마다 다름)
# Windows: PuTTY → Serial → COM포트, 115200
# Linux/Mac:
screen /dev/ttyUSB1 115200

# 보드 로그인
login: xilinx  /  password: xilinx

# IP 확인
ip addr show eth0

# IP 고정 안 되어있으면 수정
sudo vi /etc/network/interfaces
# 또는
sudo vi /etc/systemd/network/eth0.network
```
### [2] Bitstream 준비 
(ssh 연결후)
1. 166.104.144.145 Host 서버 접속 -> vivado들어가면 -zcu111이 자동으로 붙어있어서 여기서 PL을 vivado로 개발해서  
 → Generate Bitstream → .bit + .hwh 파일 생성 → (jtag로 bitstream을 보드에 업로드)

2. .bit + .hwh 파일 (+ ps용 코드) 이미 있으면 scp 로 파일 옮겨놓음됨.
```
# 보드에 폴더 생성
ssh xilinx@166.104.140.13 "mkdir -p /home/xilinx/layernorm"

# 파일 복사
scp zcu111_ps_pynq.py          xilinx@166.104.140.13:/home/xilinx/layernorm/
scp FPGA_block_design/layernorm_HIL.bit  xilinx@166.104.140.13:/home/xilinx/layernorm/
scp FPGA_block_design/layernorm_HIL.hwh  xilinx@166.104.140.13:/home/xilinx/layernorm/
```
### [3] 실행

1. 166.104.144.145 Host 서버 접속 + 가상환경(docker,conda,..) 셋팅 : 터미널1 (145서버)
```
conda create -n bert_hw python=3.10
conda activate bert_hw
```
2. custom transformer src 코드로 실행되게 함 : 터미널1 (145서버)
```
cd transformer
pip install -e . //global python에서 하지 않도록 주의!
cd src
```

3. ps에서  zcu111_ps_pynq.py 실행 : 터미널 2 (145서버 → 보드 SSH)
```
ssh xilinx@166.104.140.13
python3 /home/xilinx/layernorm/zcu111_ps_pynq.py

= TCP가 blocking으로 유지됨
```
4. Host PC에서 run_glue.py 실행 : 터미널1 (145서버)  
```
run_glue_FXP_sst2.sh
```

**< layernorm HW MODE 설명 >** 
1. Mode 1: HW 출력이 다음 레이어로 전달 (실제 HW accuracy)
```
python3 run_glue.py \
  --model_name_or_path ModelTC/bert-base-uncased-sst2 \
  --task_name sst2 --do_eval \
  --max_seq_length 128 \
  --layernorm_method hw_mode1 \
  --hw_ip 166.104.140.13  --hw_port 5000 \
  --output_dir ./results/
```
2. Mode 2: SW 계속 흐름 + HW 오차 비교
```
python3 run_glue.py ... --layernorm_method hw_mode2 ...
```
---

# 진행 흐름
```
Host PC (custom_norm.py)
  float32 (B, seq_len, d_model) → × 256  → int16 row-major
    for b in range(B):
    배치1개
        → TCP 헤더 [seq_len|d_model] + 데이터 전송
                            ↓
                  ZCU111 PS (zcu111_ps_pynq.py)
                    row-major int16 수신
                    MODULE_NUM=8 단위로 분할
                    chunk.T.flatten() → column-first
                    DMA ITER1 → ITER2
                    reshape → row-major
                    TCP 결과 전송
                            ↓
        
Host PC (custom_norm.py)
        outputs.append

  int16 row-major (B, seq_len, d_model) → ÷ 256.0 → float32
  
  weight * normalized + bias
  → 다음 레이어로
  ```
