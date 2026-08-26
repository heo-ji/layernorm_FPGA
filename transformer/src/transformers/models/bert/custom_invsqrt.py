"""
custom_invsqrt.py 교체 (layernorm_FPGA/transformer/src/transformers/models/bert/custom_invsqrt.py)


##달라진 점 (__init__의 LUT 로딩부만)

원본은 raw CSV(inv_SQRT_d.csv 등, 커브핏 원본 float)를 header 처리 없이
pd.read_csv()로 읽어서 (그래서 첫 줄이 컬럼 헤더로 먹혀 인덱스가 한 칸씩 밀림)
torch.floor(value*scale)/scale 로 직접 양자화한다.

근데 실제 RTL(invsqrt_LUT.v)과 이미 양자화되어 있는 _fixed CSV
(inv_SQRT_d_fixed.csv 등)는 floor가 아니라 반올림(round-to-nearest)으로
만들어져 있다 — 24*3=72개 항목을 RTL hex와 전수 비교해서 71개가 정확히
"floor와 round가 갈리는 지점에서는 항상 round 쪽" 패턴으로 확인됨

그래서 이 파일은
  1. header=None 추가 (인덱스 밀림 방지)
  2. raw CSV 대신 _fixed CSV를 읽는다 (이미 RTL과 100% 일치하는 정수값의
     hex 문자열이라, floor/round 양자화를 아예 할 필요가 없어짐 — 파싱만 하면 됨)

forward()는 원본과 완전히 동일 — self.d/self.s/self.t 텐서의 "내용"만 정확해지는 것뿐, 룩업/보간 로직 자체는 안 바뀐다.
"""

import torch
import numbers
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules._functions import CrossMapLRN2d as _cross_map_lrn2d
from torch.nn import functional as F
from torch.nn import init

from torch import Tensor, Size
from typing import Union, List, Tuple

import pandas as pd
import os


def _parse_verilog_hex(raw: str, bits: int) -> int:
    """"16'sh80_00" 또는 "00_0487" 같은 문자열을 bits폭 2의 보수 정수로 변환.
    "16'sh80_00" -> "'" 뒤 "sh80_00" -> 앞의 s/h 라디스 문자 제거 -> "80_00"
    -> "_" 제거 -> "8000" -> int(...,16)=32768 -> bit(bits-1) 켜져있으니 음수로 변환."""
    s = raw.strip()
    if "'" in s:
        s = s.split("'")[-1]
        s = s.lstrip("shSH")
    s = s.replace("_", "")
    val = int(s, 16)
    if val >= (1 << (bits - 1)):
        val -= (1 << bits)
    return val


def _parse_d_fixed(raw: str) -> float:
    """d(threshold) 테이블: 24bit Q8.16, 마지막 줄만 문자열 "inf"."""
    s = str(raw).strip()
    if s.lower() == "inf":
        return float("inf")
    return _parse_verilog_hex(s, 24) / 65536.0


def _parse_st_fixed(raw: str) -> float:
    """slope/intercept 테이블: 16bit Q8.8, "16'sh.." 형식."""
    return _parse_verilog_hex(str(raw), 16) / 256.0


class custom_invsqrt(Module):
    def __init__(self):
        super().__init__()

        base_path = os.path.dirname(__file__)  # 현재 .py 파일이 있는 위치
        d_csv_path = os.path.join(base_path, "invsqrt_csv_file/inv_SQRT_d_fixed.csv")
        s_csv_path = os.path.join(base_path, "invsqrt_csv_file/inv_SQRT_s_fixed.csv")
        t_csv_path = os.path.join(base_path, "invsqrt_csv_file/inv_SQRT_t_fixed.csv")

        d_raw = pd.read_csv(d_csv_path, header=None)
        s_raw = pd.read_csv(s_csv_path, header=None)
        t_raw = pd.read_csv(t_csv_path, header=None)

        # _fixed CSV는 이미 RTL과 bit-exact한 정수값을 hex 문자열로 인코딩해둔 것이라
        # floor/round 양자화가 필요 없음 — 파싱만 하면 됨.
        d = [_parse_d_fixed(v) for v in d_raw.iloc[:, 0]]
        s = [_parse_st_fixed(v) for v in s_raw.iloc[:, 0]]
        t = [_parse_st_fixed(v) for v in t_raw.iloc[:, 0]]

        self.register_buffer('d', torch.tensor(d, dtype=torch.float32))
        self.register_buffer('s', torch.tensor(s, dtype=torch.float32))
        self.register_buffer('t', torch.tensor(t, dtype=torch.float32))

    def forward(self, input):
        x = input
        self.d = self.d.to(x.device)
        self.s = self.s.to(x.device)
        self.t = self.t.to(x.device)

        idx = torch.searchsorted(self.d, x)
        # 인덱스가 배열의 크기를 벗어나지 않도록 클램핑
        idx_clamped = torch.clamp(idx, 0, self.s.size(0) - 1)

        #lut연산
        s_multiply_x = self.s[idx_clamped] * x  ## (8.8)*(8.16)

        scale_factor_8 = 2**8

        s_multiply_x = torch.floor(s_multiply_x * scale_factor_8)/scale_factor_8 #소수부8
        s_multiply_x = torch.clip(s_multiply_x, -2**7, 2**7 - 1/scale_factor_8) #정수부8.소수부8


        result =  s_multiply_x + self.t[idx_clamped] ## (8.8)+8.8
        return result
