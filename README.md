```
[e2e-bert-accel-SW] repository의 transformers과 동일


layernorm_FPGA\transformer\
├── setup.py
├── pyproject.toml
└── src\
    ├── transformers\       ← 전체 (custom 파일 포함)
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