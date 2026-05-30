# layernorm_FPGA
fpga환경에서 HIL 환경 구축
# LayerNorm HIL (Hardware-in-the-Loop) 실행

[ Project : End-to-end transformer encoder FPGA accelerator ]
1. **SW** : [`repository [e2e-bert-accel-SW]`](https://github.com/heo-ji/e2e-bert-accel-SW)
2. **layernorm HW** : [`repository [layernorm_HW]`](https://github.com/heo-ji/layernorm_HW)
    - IP 코어 (데이터 포맷, 비트폭, 파라미터) 설명 [`링크`](./doc/README_HW.md)
    - AXI wrapper (레지스터 맵, FSM, 제어/데이터전송 시퀀스) 설명 [`링크`](./doc/README_AXI_WRAPPER.md)
    - HIL 환경 overview (Host PC ↔ ZCU111 board ) 설명 [`링크`](./doc/README_FPGA_overview.md)

---

```
[e2e-bert-accel-SW] repository의 transformers과 동일


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
cd transformer
pip install -e . //global python에서 하지 않도록 주의!
cd src
bash run_glue_FXP_sst2.sh
```
---

zcu111보드의 pynq버전 2.1이상이면
PYNQ에서는 .bit + .hwh 두 개 (파일명 base name이 같게!) 옮겨놓는다.

layernorm_FPGA\FPGA_block_design\
├── layernorm_HIL.bit
├── layernorm_HIL.hwh

```

