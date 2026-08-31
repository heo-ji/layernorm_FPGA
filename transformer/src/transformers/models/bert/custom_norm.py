"""
torch.nn.modules.normalization에서 복붙
class Custom_LayerNorm
"""

from .custom_invsqrt import custom_invsqrt #추가함

# ──────────────────────────────────────────────────────────────────
# HW LayerNorm TCP 클라이언트 (ZCU111 PS와 통신)
# ──────────────────────────────────────────────────────────────────
import socket
import struct
import numpy as np

def _recv_exact(conn, n: int) -> bytes:
    """TCP 소켓에서 정확히 n 바이트 수신."""
    buf = b''
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("ZCU111 연결이 끊어짐")
        buf += chunk
    return buf


class HWLayerNormClient:
    """
    ZCU111 PS와 TCP 통신하는 싱글톤 클라이언트.

    사용법 (run_glue.py 에서 한 번만 호출):
        HWLayerNormClient.connect(ip='166.104.140.13', port=5000)

    프로토콜:
        송신: [seq_len(uint16 LE)][d_model(uint16 LE)] + seq_len*d_model*int16 (row-major)
        수신: seq_len*d_model*int16 (row-major)
        * column-first 변환은 PS(zcu111_ps_pynq.py)에서 처리
    """
    _instance = None

    @classmethod
    def connect(cls, ip: str, port: int):
        """서버에 연결. run_glue.py 에서 inference 시작 전 1회 호출."""
        inst = cls.__new__(cls)
        inst.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        inst.sock.connect((ip, port))
        inst.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        cls._instance = inst
        print(f"[HWLayerNormClient] ZCU111 연결 완료: {ip}:{port}")
        return inst

    @classmethod
    def get(cls):
        if cls._instance is None:
            raise RuntimeError("HWLayerNormClient.connect() 를 먼저 호출하세요.")
        return cls._instance

    def run(self, x_np: np.ndarray) -> np.ndarray:
        """
        TCP 송수신만 담당.

        x_np   : (seq_len, d_model)  float (값이 이미 ×256 클리핑된 상태)
        return : (seq_len, d_model)  int16 numpy
        """
        seq_len, d_model = x_np.shape

        # 호출 측에서 이미 ×256 + clip 완료 → int16 캐스팅도 앞에서함
        int16_data = x_np #.astype(np.int16)

        # 헤더: seq_len, d_model (uint16 little-endian)
        header = struct.pack('<HH', seq_len, d_model)

        # 송신: 헤더 + row-major int16 바이트열
        self.sock.sendall(header + int16_data.tobytes())

        # 수신: seq_len × d_model × 2 바이트 → int16 반환
        raw = _recv_exact(self.sock, seq_len * d_model * 2)
        return np.frombuffer(raw, dtype=np.int16).reshape(seq_len, d_model).copy()

    def close(self):
        self.sock.close()
        HWLayerNormClient._instance = None
# ──────────────────────────────────────────────────────────────────


import os
import json
import torch
import numbers
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn.modules._functions import CrossMapLRN2d as _cross_map_lrn2d
from torch.nn import functional as F
from torch.nn import init

from torch import Tensor, Size
from typing import Union, List, Tuple

__all__ = ['LocalResponseNorm', 'CrossMapLRN2d', 'LayerNorm', 'GroupNorm']

class LocalResponseNorm(Module):
    r"""Applies local response normalization over an input signal.

    The input signal is composed of several input planes, where channels occupy the second dimension.
    Applies normalization across channels.

    .. math::
        b_{c} = a_{c}\left(k + \frac{\alpha}{n}
        \sum_{c'=\max(0, c-n/2)}^{\min(N-1,c+n/2)}a_{c'}^2\right)^{-\beta}

    Args:
        size: amount of neighbouring channels used for normalization
        alpha: multiplicative factor. Default: 0.0001
        beta: exponent. Default: 0.75
        k: additive factor. Default: 1

    Shape:
        - Input: :math:`(N, C, *)`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> lrn = nn.LocalResponseNorm(2)
        >>> signal_2d = torch.randn(32, 5, 24, 24)
        >>> signal_4d = torch.randn(16, 5, 7, 7, 7, 7)
        >>> output_2d = lrn(signal_2d)
        >>> output_4d = lrn(signal_4d)

    """

    __constants__ = ['size', 'alpha', 'beta', 'k']
    size: int
    alpha: float
    beta: float
    k: float

    def __init__(self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1.) -> None:
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input: Tensor) -> Tensor:
        return F.local_response_norm(input, self.size, self.alpha, self.beta,
                                     self.k)

    def extra_repr(self):
        return '{size}, alpha={alpha}, beta={beta}, k={k}'.format(**self.__dict__)


class CrossMapLRN2d(Module):
    size: int
    alpha: float
    beta: float
    k: float

    def __init__(self, size: int, alpha: float = 1e-4, beta: float = 0.75, k: float = 1) -> None:
        super().__init__()
        self.size = size
        self.alpha = alpha
        self.beta = beta
        self.k = k

    def forward(self, input: Tensor) -> Tensor:
        return _cross_map_lrn2d.apply(input, self.size, self.alpha, self.beta,
                                      self.k)

    def extra_repr(self) -> str:
        return '{size}, alpha={alpha}, beta={beta}, k={k}'.format(**self.__dict__)


_shape_t = Union[int, List[int], Size]


# class LayerNorm(Module):
#     r"""Applies Layer Normalization over a mini-batch of inputs.

#     This layer implements the operation as described in
#     the paper `Layer Normalization <https://arxiv.org/abs/1607.06450>`__

#     .. math::
#         y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

#     The mean and standard-deviation are calculated over the last `D` dimensions, where `D`
#     is the dimension of :attr:`normalized_shape`. For example, if :attr:`normalized_shape`
#     is ``(3, 5)`` (a 2-dimensional shape), the mean and standard-deviation are computed over
#     the last 2 dimensions of the input (i.e. ``input.mean((-2, -1))``).
#     :math:`\gamma` and :math:`\beta` are learnable affine transform parameters of
#     :attr:`normalized_shape` if :attr:`elementwise_affine` is ``True``.
#     The standard-deviation is calculated via the biased estimator, equivalent to
#     `torch.var(input, unbiased=False)`.

#     .. note::
#         Unlike Batch Normalization and Instance Normalization, which applies
#         scalar scale and bias for each entire channel/plane with the
#         :attr:`affine` option, Layer Normalization applies per-element scale and
#         bias with :attr:`elementwise_affine`.

#     This layer uses statistics computed from input data in both training and
#     evaluation modes.

#     Args:
#         normalized_shape (int or list or torch.Size): input shape from an expected input
#             of size

#             .. math::
#                 [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
#                     \times \ldots \times \text{normalized\_shape}[-1]]

#             If a single integer is used, it is treated as a singleton list, and this module will
#             normalize over the last dimension which is expected to be of that specific size.
#         eps: a value added to the denominator for numerical stability. Default: 1e-5
#         elementwise_affine: a boolean value that when set to ``True``, this module
#             has learnable per-element affine parameters initialized to ones (for weights)
#             and zeros (for biases). Default: ``True``.
#         bias: If set to ``False``, the layer will not learn an additive bias (only relevant if
#             :attr:`elementwise_affine` is ``True``). Default: ``True``.

#     Attributes:
#         weight: the learnable weights of the module of shape
#             :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
#             The values are initialized to 1.
#         bias:   the learnable bias of the module of shape
#                 :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
#                 The values are initialized to 0.

#     Shape:
#         - Input: :math:`(N, *)`
#         - Output: :math:`(N, *)` (same shape as input)

#     Examples::

#         >>> # NLP Example
#         >>> batch, sentence_length, embedding_dim = 20, 5, 10
#         >>> embedding = torch.randn(batch, sentence_length, embedding_dim)
#         >>> layer_norm = nn.LayerNorm(embedding_dim)
#         >>> # Activate module
#         >>> layer_norm(embedding)
#         >>>
#         >>> # Image Example
#         >>> N, C, H, W = 20, 5, 10, 10
#         >>> input = torch.randn(N, C, H, W)
#         >>> # Normalize over the last three dimensions (i.e. the channel and spatial dimensions)
#         >>> # as shown in the image below
#         >>> layer_norm = nn.LayerNorm([C, H, W])
#         >>> output = layer_norm(input)

#     .. image:: ../_static/img/nn/layer_norm.jpg
#         :scale: 50 %

#     """

#     __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
#     normalized_shape: Tuple[int, ...]
#     eps: float
#     elementwise_affine: bool

#     def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, elementwise_affine: bool = True,
#                  bias: bool = True, device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super().__init__()
#         if isinstance(normalized_shape, numbers.Integral):
#             # mypy error: incompatible types in assignment
#             normalized_shape = (normalized_shape,)  # type: ignore[assignment]
#         self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
#         self.eps = eps
#         self.elementwise_affine = elementwise_affine
#         if self.elementwise_affine:
#             self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
#             if bias:
#                 self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
#             else:
#                 self.register_parameter('bias', None)
#         else:
#             self.register_parameter('weight', None)
#             self.register_parameter('bias', None)

#         self.reset_parameters()
#         self.invsqrt = custom_invsqrt() ##custom

#     def reset_parameters(self) -> None:
#         if self.elementwise_affine:
#             init.ones_(self.weight)
#             if self.bias is not None:
#                 init.zeros_(self.bias)

    
#     def forward(self, input: Tensor) -> Tensor:
#         dmodel = input.size(2)
        
        
#         scale_factor_8 = 2**8
#         scale_factor_10 = 2**10
#         scale_factor_16 = 2**16

#         #input = 8.8
#         input_fx16 = torch.floor(input * scale_factor_8)/scale_factor_8 #소수부8
#         input_fx16 = torch.clip(input_fx16, -2**7, 2**7 - 1/scale_factor_8) #정수부8.8
#         #torch.save(input_fx16,'/home/user/HJH/transformers/src/MPWnormfile/input_8_8.pt') ##저장

#         acc_sum = torch.sum(input_fx16, dim=-1, keepdim=True)
#         #acc_sum = 26bit(18.8)
#         acc_sum = torch.floor(acc_sum * scale_factor_8)/scale_factor_8 #??필요한가? #소수부8
#         acc_sum = torch.clip(acc_sum, -2**17, 2**17- 1)#??필요한가? 정수부18
#         #torch.save(acc_sum,'/home/user/HJH/transformers/src/MPWnormfile/acc_sum_18_8.pt') ##저장

#         #mean계산
#         mean = acc_sum /2**8 #/dmodel
#         mean = torch.floor(mean * scale_factor_8)/scale_factor_8 #소수부8
#         mean = mean *0.33203125 #18.8*0.8 = 18.16

#         #mean = 16bit(8.8)로 만들기
#         mean = torch.clip(mean, -2**7, 2**7 - 1/scale_factor_8) #정수부8.8
#         mean = torch.floor(mean * scale_factor_8)/scale_factor_8 #소수부8
#         #torch.save(mean,'/home/user/HJH/transformers/src/MPWnormfile/mean_8_8.pt') ##저장

#         #분산계산
#         #X^2 = 32bit(16.16) (8.8의 제곱)
#         x2 = input_fx16*input_fx16
#         #X^2 = 24bit(16.8)
#         # x2 = torch.clip(x2, -2**15, 2**15- 1)#??필요한가? 정수부16
#         x2 = torch.floor(x2 * scale_factor_8)/scale_factor_8 #소수부8 #rescaling and round
        
#         #acc_sum_x2= 34bit(26.8 )
#         acc_sum_x2 = torch.sum(x2, dim=-1, keepdim=True)
#         acc_sum_x2 = torch.floor(acc_sum_x2 * scale_factor_8)/scale_factor_8 #소수부8
#         #torch.save(acc_sum_x2,'/home/user/HJH/transformers/src/MPWnormfile/acc_sum_x2_26_8.pt') ##저장
        
        
#         #mean_x2
#         mean_x2 = acc_sum_x2 /2**8 #/dmodel (26.8)>>8 = 16.8
#         mean_x2 = torch.floor(mean_x2 * scale_factor_8)/scale_factor_8 #소수부8
#         mean_x2 = mean_x2 *0.33203125 #18.8*0.8 = 18.16

#         ###mean_x2 = 26bit(16.10)                                 
#         ###mean_x2 = torch.floor(mean_x2 * scale_factor_10)/scale_factor_10 #소수부10

#         #mean_x2 = 32bit(16.16)   
#         mean_x2 = torch.floor(mean_x2 * scale_factor_16)/scale_factor_16 #소수부16
#         mean_x2 = torch.clip(mean_x2, -2**15, 2**15 - 1/scale_factor_16) #정수부16.16    
#         #torch.save(mean_x2,'/home/user/HJH/transformers/src/MPWnormfile/mean_x2_16_16.pt') ##저장 

#         #E(x)^2 = 32bit(16.16)
#         temp = mean*mean
#         temp = torch.floor(temp * scale_factor_16)/scale_factor_16 #소수부16

#         ###E(x)^2 = 32bit(16.16)  -> 26bit(16.10)로 만들기 
#         ###temp = torch.floor(temp * scale_factor_10)/scale_factor_10 #소수부10
#         temp = torch.clip(temp, -2**15, 2**15 - 1/scale_factor_16) #정수부16.16
#         ###var = 26bit(16.10) - 26bit(16.10)


#         #var = 32bit - 32bit = 32bit(16.16)  -> (8.16)
#         var = mean_x2 - temp

#         var = torch.floor(var * scale_factor_16)/scale_factor_16 #소수부16
#         var = torch.clip(var, -2**7, 2**7 - 1/scale_factor_16) #정수부 (8.16)
#         #torch.save(var,'/home/user/HJH/transformers/src/MPWnormfile/var_8_16.pt') ##저장 

#         eps = self.eps #(.16)
#         eps = round(eps * scale_factor_16)/scale_factor_16 #소수부16 = eps=0.0이됨
#         eps = 0.0000152587890625

#         v=var + eps
#         # invsqrt = 1/ torch.sqrt(var + eps)
#         invsqrt = self.invsqrt(v) #custom_invsqrt.py

#         #inverse square root = 8.8
#         invsqrt = torch.floor(invsqrt * scale_factor_8)/scale_factor_8
#         invsqrt = torch.clip(invsqrt, -2**7, 2**7 - 1/scale_factor_8)
#         #torch.save(invsqrt,'/home/user/HJH/transformers/src/MPWnormfile/invsqrt_8_8.pt') ##저장 

#         # 8.8  * 8.8 = 16.16
#         normalized = (input_fx16 - mean) * invsqrt 
#         #normalized = 8.8
#         normalized = torch.floor(normalized * scale_factor_8)/scale_factor_8 #소수부8
#         normalized = torch.clip(normalized, -2**7, 2**7 - 1/scale_factor_8) #정수부8
#         #torch.save(normalized,'/home/user/HJH/transformers/src/MPWnormfile/normalized_8_8.pt') ##저장    

#         if self.weight is not None:
#             #self.weight = 8.8
#             weight = torch.floor(self.weight * scale_factor_8)/scale_factor_8 #소수부8
#             weight = torch.clip(weight, -2**7, 2**7 - 1/scale_factor_8) #정수부8

#             #torch.save(weight,'/home/user/HJH/transformers/src/MPWnormfile/weight_8_8.pt') ##저장  
#             out = normalized * weight
        
#         else : out = normalized
#         #8.8
#         out = torch.floor(out * scale_factor_8)/scale_factor_8 #소수부8
#         out = torch.clip(out, -2**7, 2**7 - 1/scale_factor_8) #정수부8
#         #torch.save(out,'/home/user/HJH/transformers/src/MPWnormfile/weightedout_8_8.pt') ##저장 


#         if self.bias is not None:
#             #self.bias = 8.8
#             bias = torch.floor(self.bias * scale_factor_8)/scale_factor_8 #소수부8
#             bias = torch.clip(bias, -2**7, 2**7 - 1/scale_factor_8) #정수부8

#             #torch.save(bias,'/home/user/HJH/transformers/src/MPWnormfile/bias_8_8.pt') ##저장  
#             out = out + bias

#         #8.8
#         out = torch.floor(out * scale_factor_8)/scale_factor_8 #소수부8
#         out = torch.clip(out, -2**7, 2**7 - 1/scale_factor_8) #정수부8

#         #torch.save(out,'/home/user/HJH/transformers/src/MPWnormfile/biasedout_8_8.pt') ##저장 
#         #(8.8)
#         return out
#         #중단점 23= FFN

#         # return F.layer_norm(
#         #     input, self.normalized_shape, self.weight, self.bias, self.eps)

#     def extra_repr(self) -> str:
#         return '{normalized_shape}, eps={eps}, ' \
#             'elementwise_affine={elementwise_affine}'.format(**self.__dict__)

class Custom_LayerNorm(Module):
    r"""Applies Layer Normalization over a mini-batch of inputs.
    """

    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: float
    elementwise_affine: bool

    # ── SAIF profiling pass1용 클래스 변수 (모든 Custom_LayerNorm 인스턴스가 공유) ──
    _saif_forward_idx = -1   # 전체 forward(batch) 카운터. layer0/atten 호출 시 +1
    _saif_log = {}           # {(layer_idx, block_type): {forward_idx: activity(32,)}}

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, method: str = 'original', elementwise_affine: bool = True,
                 bias: bool = True, device=None, dtype=None,
                 layer_idx: int = None, block_type: str = None, task_name: str = None,
                 tensor_save_dir: str = 'GLUEtask_tensor') -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            # mypy error: incompatible types in assignment
            normalized_shape = (normalized_shape,)  # type: ignore[assignment]
        self.normalized_shape = tuple(normalized_shape)  # type: ignore[arg-type]
        self.eps = eps
        

        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            if bias:
                self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
            else:
                self.register_parameter('bias', None)
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

        #추가함
        self.invsqrt = custom_invsqrt()
        self.method = method
        self.D_s = 384  # example value for the number of significant dimensions
        self.N_t = 8    # sampling rate for trivial dimensions

        # HW mode2 : worst-case 입력 텐서 저장용
        self._worst_case_error = -1.0
        self._worst_case_input = None

        # forward_fxp88 중간 텐서 저장용 (layer#/atten,ffn 구분 → 덮어쓰기 방지)
        self.layer_idx = layer_idx
        self.block_type = block_type
        self.task_name = task_name or 'unknown_task'
        self.tensor_save_dir = tensor_save_dir

    def _save_fxp_tensor(self, tensor: Tensor, name: str) -> None:
        """{tensor_save_dir}/layer{layer_idx}_{block_type}_{name}.pt 로 저장 (레이어/블록별로 별도 파일)."""
        save_dir = self.tensor_save_dir
        os.makedirs(save_dir, exist_ok=True)
        layer_tag = f"layer{self.layer_idx}" if self.layer_idx is not None else "layerX"
        block_tag = self.block_type or "unknown"
        save_path = os.path.join(save_dir, f"{layer_tag}_{block_tag}_{name}.pt")
        torch.save(tensor.detach().cpu(), save_path)

    # ── SAIF profiling pass1 제어용 classmethod ─────────────────────
    @classmethod
    def saif_reset(cls):
        """pass1 시작 전 호출. forward 카운터와 activity 로그 초기화."""
        cls._saif_forward_idx = -1
        cls._saif_log = {}

    @classmethod
    def saif_write_pass1_report(cls, task_name: str, report_dir: str = 'saif_convergence_report', epsilon: float = 0.0005):
        """
        pass1(전체 eval) 종료 후 호출.
        위치별(layer_idx, block_type)로 forward를 몇 개까지 누적해야
        running-mean activity가 전체 평균의 epsilon 이내로 수렴해서
        이후 다시 벗어나지 않는지(convergence_forward_count = K)를 계산하고,
        참고용으로 전체 평균에 가장 가까운 실제 forward(medoid)도 같이 기록해서
        {report_dir}/{task_name}_convergence.json 으로 저장.
        """
        report = {}
        for key, log in cls._saif_log.items():
            layer_idx, block_type = key
            forward_ids = sorted(log.keys())
            vectors = torch.stack([log[i] for i in forward_ids])  # (N, 32)
            n_total = vectors.shape[0]

            counts = torch.arange(1, n_total + 1, dtype=vectors.dtype).unsqueeze(1)
            running_mean = torch.cumsum(vectors, dim=0) / counts  # (N, 32)
            final_mean = running_mean[-1]

            dist_to_final = torch.sqrt(torch.mean((running_mean - final_mean) ** 2, dim=1))  # (N,)

            # K = 이 시점부터는 계속 epsilon 이내에 머무는 가장 작은 forward 개수
            last_bad = -1
            for n in range(n_total):
                if dist_to_final[n].item() > epsilon:
                    last_bad = n
            k = last_bad + 2 if last_bad >= 0 else 1

            # medoid : 참고용, 전체 평균에 가장 가까운 실제 forward
            dist_per_forward = torch.sqrt(torch.mean((vectors - final_mean) ** 2, dim=1))
            best_pos = int(torch.argmin(dist_per_forward))

            checkpoints = sorted({2**p for p in range(0, 20) if 2**p <= n_total} | {n_total})

            report[f"layer{layer_idx}_{block_type}"] = {
                "num_forwards": n_total,
                "convergence_forward_count": k,
                "epsilon": epsilon,
                "medoid_forward_idx": forward_ids[best_pos],
                "medoid_distance": float(dist_per_forward[best_pos]),
                "dist_to_final_by_forward_count": {
                    str(n): float(dist_to_final[n - 1]) for n in checkpoints
                },
            }

        os.makedirs(report_dir, exist_ok=True)
        save_path = os.path.join(report_dir, f"{task_name}_convergence.json")
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        return report, save_path

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            init.ones_(self.weight)
            if self.bias is not None:
                init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        if self.method == 'original':
            return self.forward_original(input)
        elif self.method == 'custom_invsqrt_norm':
            return self.forward_fxp88(input)
        elif self.method == 'dualpath_norm':
            return self.forward_dual_path(input)
        elif self.method == 'hw_mode1':
            return self.forward_hw_mode1(input)
        elif self.method == 'hw_mode2':
            return self.forward_hw_mode2(input)
        elif self.method == 'profiling_pass1':
            return self.forward_profiling_pass1(input)
        elif self.method == 'profiling_pass2':
            return self.forward_profiling_pass2(input)
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    # ── HW Mode 1 : HW 출력을 다음 레이어로 전달 (실제 HW accuracy 측정) ──
    def forward_hw_mode1(self, input: Tensor) -> Tensor:
        """
        HW LayerNorm 결과를 그대로 사용 → HW 오차가 모델 전체에 누적됨.
        최종 accuracy = 실제 HW accuracy.
        """
        
        client = HWLayerNormClient.get()
        B = input.shape[0]

        # 배치 전체 float32 → int16 스케일 변환
        input_int16 = torch.clip(input * 256, -32768, 32767)

        ################# TCP -> FPGA(PS)##############################################
        outputs = []
        for b in range(B):
            x_np   = input_int16[b].detach().cpu().numpy()   # (seq_len, d_model)
            hw_out = client.run(x_np)                         # (seq_len, d_model) int16
            outputs.append(torch.from_numpy(hw_out))
        #############################################################################
        # 배치 전체 int16 → float32 복원 (normalized, weight/bias 미적용)
        normalized = torch.stack(outputs, dim=0).to(input.device).float() / 256.0

        # ── weight 8.8 포맷 적용 ──────────────────────────────
        scale_factor_8 = 2**8
        if self.weight is not None:
            weight = torch.floor(self.weight * scale_factor_8) / scale_factor_8  # 소수부 8
            weight = torch.clip(weight, -2**7, 2**7 - 1/scale_factor_8)          # 정수부 8
            out = normalized * weight
        else:
            out = normalized
        out = torch.floor(out * scale_factor_8) / scale_factor_8
        out = torch.clip(out, -2**7, 2**7 - 1/scale_factor_8)

        # ── bias 8.8 포맷 적용 ───────────────────────────────
        if self.bias is not None:
            bias = torch.floor(self.bias * scale_factor_8) / scale_factor_8      # 소수부 8
            bias = torch.clip(bias, -2**7, 2**7 - 1/scale_factor_8)              # 정수부 8
            out = out + bias
        out = torch.floor(out * scale_factor_8) / scale_factor_8
        out = torch.clip(out, -2**7, 2**7 - 1/scale_factor_8)
        return out

    # ── HW Mode 2 : SW 값으로 계속 흐르되, HW vs SW 오차 비교 + worst-case 저장 ──
    def forward_hw_mode2(self, input: Tensor) -> Tensor:
        """
        SW golden 결과로 모델이 계속 흐름 → accuracy 영향 없음.
        오차가 최대인 입력 텐서를 저장.
        """
        # SW golden 계산 (weight/bias 포함 전체 결과)
        #sw_out = self.forward_fxp88(input)
        dmodel = input.size(2)
        
        scale_factor_8 = 2**8
        scale_factor_16 = 2**16

        #input = 8.8
        input_fx16 = torch.floor(input * scale_factor_8)/scale_factor_8 #소수부8
        input_fx16 = torch.clip(input_fx16, -2**7, 2**7 - 1/scale_factor_8) #정수부8.8
        #torch.save(input_fx16,'/home/user/HJH/transformers/src/MPWnormfile/input_8_8.pt') ##저장

        acc_sum = torch.sum(input_fx16, dim=-1, keepdim=True)
        #acc_sum = 26bit(18.8)
        acc_sum = torch.floor(acc_sum * scale_factor_8)/scale_factor_8 ##소수부8
        acc_sum = torch.clip(acc_sum, -2**17, 2**17- 1/scale_factor_8)#정수부18 .소수부8
        #torch.save(acc_sum,'/home/user/HJH/transformers/src/MPWnormfile/acc_sum_18_8.pt') ##저장

        #mean계산
        mean = acc_sum /2**8 #Q(18.8) -> Q(10.8)
        mean = torch.floor(mean * scale_factor_8)/scale_factor_8 #소수부8로 precision 맞춤
        mean = mean *0.33203125 #Q(10.8)*0.8 = 10.16

        #mean = 16bit(8.8)로 만들기 = saturation
        mean = torch.clip(mean, -2**7, 2**7 - 1/scale_factor_8) #정수부8.8
        mean = torch.floor(mean * scale_factor_8)/scale_factor_8 #소수부8
        #torch.save(mean,'/home/user/HJH/transformers/src/MPWnormfile/mean_8_8.pt') ##저장

        #분산계산
        #X^2 = 32bit(16.16) (8.8의 제곱)
        x2 = input_fx16*input_fx16
        #X^2 = 24bit(16.8)
        
        x2 = torch.floor(x2 * scale_factor_8)/scale_factor_8 #소수부8 #rescaling and round
        #[try]#x2 = torch.clip(x2, -2**15, 2**15- 1/scale_factor_8)# 정수부16.8

        #acc_sum_x2= 34bit(26.8 )
        acc_sum_x2 = torch.sum(x2, dim=-1, keepdim=True)
        acc_sum_x2 = torch.floor(acc_sum_x2 * scale_factor_8)/scale_factor_8 #소수부8
        #[try]#acc_sum_x2 = torch.clip(acc_sum_x2, -2**25, 2**25- 1/scale_factor_8)# 정수부26.8
        #torch.save(acc_sum_x2,'/home/user/HJH/transformers/src/MPWnormfile/acc_sum_x2_26_8.pt') ##저장
        
        
        #mean_x2
        mean_x2 = acc_sum_x2 /2**8 #/dmodel (26.8)>>8 = 18.8
        mean_x2 = torch.floor(mean_x2 * scale_factor_8)/scale_factor_8 #소수부8
        mean_x2 = mean_x2 *0.33203125 #18.8*0.8 = 18.16

        ###mean_x2 = 26bit(16.10)                                 
        ###mean_x2 = torch.floor(mean_x2 * scale_factor_10)/scale_factor_10 #소수부10

        #mean_x2 = 32bit(16.16)   
        mean_x2 = torch.floor(mean_x2 * scale_factor_16)/scale_factor_16 #소수부16
        mean_x2 = torch.clip(mean_x2, -2**15, 2**15 - 1/scale_factor_16) #정수부16.16    
        #torch.save(mean_x2,'/home/user/HJH/transformers/src/MPWnormfile/mean_x2_16_16.pt') ##저장 

        #E(x)^2 = 32bit(16.16)
        temp = mean*mean
        temp = torch.floor(temp * scale_factor_16)/scale_factor_16 #소수부16
        temp = torch.clip(temp, -2**15, 2**15 - 1/scale_factor_16) #정수부16.16
        
        ###E(x)^2 = 32bit(16.16)  -> 26bit(16.10)로 만들기 
        ###temp = torch.floor(temp * scale_factor_10)/scale_factor_10 #소수부10
        ###var = 26bit(16.10) - 26bit(16.10)


        #var = 32bit - 32bit = 32bit(16.16)  -> (8.16)로 saturation
        var = mean_x2 - temp
        var = torch.floor(var * scale_factor_16)/scale_factor_16 #소수부16
        var = torch.clip(var, -2**7, 2**7 - 1/scale_factor_16) #정수부 (8.16)
        #torch.save(var,'/home/user/HJH/transformers/src/MPWnormfile/var_8_16.pt') ##저장 

        eps = 0.0000152587890625

        v = var + eps
        # invsqrt = 1/ torch.sqrt(var + eps)
        invsqrt = self.invsqrt(v) #custom_invsqrt.py

        #inverse square root = 8.8
        invsqrt = torch.floor(invsqrt * scale_factor_8)/scale_factor_8
        invsqrt = torch.clip(invsqrt, -2**7, 2**7 - 1/scale_factor_8)
        
        # 8.8  * 8.8 = 16.16
        normalized = (input_fx16 - mean) * invsqrt 
        #normalized = 8.8
        normalized = torch.floor(normalized * scale_factor_8)/scale_factor_8 #소수부8
        normalized = torch.clip(normalized, -2**7, 2**7 - 1/scale_factor_8) #정수부8

        if self.weight is not None:
            #self.weight = 8.8
            weight = torch.floor(self.weight * scale_factor_8)/scale_factor_8 #소수부8
            weight = torch.clip(weight, -2**7, 2**7 - 1/scale_factor_8) #정수부8
            out = normalized * weight
        
        else : out = normalized
        #8.8
        out = torch.floor(out * scale_factor_8)/scale_factor_8 #소수부8
        out = torch.clip(out, -2**7, 2**7 - 1/scale_factor_8) #정수부8

        if self.bias is not None:
            #self.bias = 8.8
            bias = torch.floor(self.bias * scale_factor_8)/scale_factor_8 #소수부8
            bias = torch.clip(bias, -2**7, 2**7 - 1/scale_factor_8) #정수부8
            out = out + bias

        #8.8
        out = torch.floor(out * scale_factor_8)/scale_factor_8 #소수부8
        out = torch.clip(out, -2**7, 2**7 - 1/scale_factor_8) #정수부8

        sw_out = out


        client = HWLayerNormClient.get()
        B = input.shape[0]

        # 배치 전체 float32 → int16 스케일 변환 + int16 변환
        input_scale = torch.clip(input * 256, -32768, 32767)
        input_int16 = input_scale.to(torch.int16)

        ################# TCP -> FPGA(PS)##############################################

        for b in range(B):
            x_np   = input_int16[b].detach().cpu().numpy()
            hw_out = client.run(x_np)                          # int16, normalization만 (weight/bias 미적용)
            hw_fp  = hw_out.astype(np.float32) / 256.0        # float32 복원 (normalized 수준)

            # 비교: HW normalized vs SW normalized
            sw_np   = normalized[b].detach().cpu().float().numpy()
            max_err = float(np.abs(hw_fp - sw_np).max())

            if max_err > self._worst_case_error:
                self._worst_case_error = max_err
                self._worst_case_input = x_np.copy()
                print(f"[HW Mode2] worst-case 갱신: max_err={max_err:.5f}")

        return sw_out

    # ── SAIF profiling pass1 : 위치별 activity 수집 (수렴 forward 수 K 확인용) ──
    def forward_profiling_pass1(self, input: Tensor) -> Tensor:
        """
        (layer_idx, block_type) 위치별로 16bit toggle rate + static probability(Q8.8 기준)를
        forward마다 기록만 함. tensor 저장 없음.
        run_glue.py에서 evaluate() 종료 후 saif_write_pass1_report()를 호출해
        위치별 수렴 forward 수(K)와 참고용 medoid forward_idx를 report로 남김.
        반환값은 forward_fxp88과 동일한 SW golden 결과 (accuracy 영향 없음).
        """
        key = (self.layer_idx, self.block_type)

        # 전체 forward(batch) 카운터: 매 forward에서 항상 layer0/atten이 가장 먼저 호출되는 점을 이용
        if self.layer_idx == 0 and self.block_type == 'atten':
            Custom_LayerNorm._saif_forward_idx += 1
        forward_idx = Custom_LayerNorm._saif_forward_idx

        scale_factor_8 = 2**8
        scale_factor_16 = 2**16

        #input = 8.8 (RTL과 동일 포맷)
        input_fx16 = torch.floor(input * scale_factor_8)/scale_factor_8
        input_fx16 = torch.clip(input_fx16, -2**7, 2**7 - 1/scale_factor_8)

        # ── activity(toggle rate + static probability) 계산 ──
        q = torch.round(input_fx16 * scale_factor_8).to(torch.int32)
        u = q & 0xFFFF  # Q8.8 값의 16bit two's-complement bit pattern (RTL 표현과 동일)

        xor = u[:, :, 1:] ^ u[:, :, :-1]  # feature(col) 방향 인접 bit 변화
        toggle = torch.stack([((xor >> bit) & 1).float().mean() for bit in range(16)])
        p1 = torch.stack([((u >> bit) & 1).float().mean() for bit in range(16)])
        activity = torch.cat([toggle, p1]).detach().cpu()  # 32-dim

        Custom_LayerNorm._saif_log.setdefault(key, {})[forward_idx] = activity

        # ── 이하 forward_fxp88과 동일한 SW golden 연산 (accuracy 영향 없이 그대로 흘림) ──
        acc_sum = torch.sum(input_fx16, dim=-1, keepdim=True)
        acc_sum = torch.floor(acc_sum * scale_factor_8)/scale_factor_8
        acc_sum = torch.clip(acc_sum, -2**17, 2**17- 1/scale_factor_8)

        mean = acc_sum /2**8
        mean = torch.floor(mean * scale_factor_8)/scale_factor_8
        mean = mean *0.33203125
        mean = torch.clip(mean, -2**7, 2**7 - 1/scale_factor_8)
        mean = torch.floor(mean * scale_factor_8)/scale_factor_8

        x2 = input_fx16*input_fx16
        x2 = torch.floor(x2 * scale_factor_8)/scale_factor_8

        acc_sum_x2 = torch.sum(x2, dim=-1, keepdim=True)
        acc_sum_x2 = torch.floor(acc_sum_x2 * scale_factor_8)/scale_factor_8

        mean_x2 = acc_sum_x2 /2**8
        mean_x2 = torch.floor(mean_x2 * scale_factor_8)/scale_factor_8
        mean_x2 = mean_x2 *0.33203125
        mean_x2 = torch.floor(mean_x2 * scale_factor_16)/scale_factor_16
        mean_x2 = torch.clip(mean_x2, -2**15, 2**15 - 1/scale_factor_16)

        temp = mean*mean
        temp = torch.floor(temp * scale_factor_16)/scale_factor_16
        temp = torch.clip(temp, -2**15, 2**15 - 1/scale_factor_16)

        var = mean_x2 - temp
        var = torch.floor(var * scale_factor_16)/scale_factor_16
        var = torch.clip(var, -2**7, 2**7 - 1/scale_factor_16)

        eps = 0.0000152587890625
        v = var + eps
        invsqrt = self.invsqrt(v)
        invsqrt = torch.floor(invsqrt * scale_factor_8)/scale_factor_8
        invsqrt = torch.clip(invsqrt, -2**7, 2**7 - 1/scale_factor_8)

        normalized = (input_fx16 - mean) * invsqrt
        normalized = torch.floor(normalized * scale_factor_8)/scale_factor_8
        normalized = torch.clip(normalized, -2**7, 2**7 - 1/scale_factor_8)

        if self.weight is not None:
            weight = torch.floor(self.weight * scale_factor_8)/scale_factor_8
            weight = torch.clip(weight, -2**7, 2**7 - 1/scale_factor_8)
            out = normalized * weight
        else:
            out = normalized
        out = torch.floor(out * scale_factor_8)/scale_factor_8
        out = torch.clip(out, -2**7, 2**7 - 1/scale_factor_8)

        if self.bias is not None:
            bias = torch.floor(self.bias * scale_factor_8)/scale_factor_8
            bias = torch.clip(bias, -2**7, 2**7 - 1/scale_factor_8)
            out = out + bias

        out = torch.floor(out * scale_factor_8)/scale_factor_8
        out = torch.clip(out, -2**7, 2**7 - 1/scale_factor_8)

        return out

    # ── SAIF profiling pass2 : 아직 미구현 ──────────────────────────
    def forward_profiling_pass2(self, input: Tensor) -> Tensor:
        """
        TODO: pass1 report의 convergence_forward_count(K)를 보고 저장 방식을 정한 뒤 구현.
        - K가 작으면: 대표 forward 여러 개를 이어붙여 그대로 SAIF 대상으로 저장
        - K가 크면: medoid 1개(+low/high) 방식으로 저장
        """
        raise NotImplementedError(
            "forward_profiling_pass2: pass1 결과(convergence_forward_count)를 보고 구현 예정"
        )

    def forward_original(self, input: Tensor) -> Tensor:
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
    
    def forward_fxp88(self, input: Tensor) -> Tensor:
        dmodel = input.size(2)
        
        scale_factor_8 = 2**8
        scale_factor_10 = 2**10
        scale_factor_16 = 2**16

        #input = 8.8
        input_fx16 = torch.floor(input * scale_factor_8)/scale_factor_8 #소수부8
        input_fx16 = torch.clip(input_fx16, -2**7, 2**7 - 1/scale_factor_8) #정수부8.8
        # self._save_fxp_tensor(input_fx16, 'input')

        acc_sum = torch.sum(input_fx16, dim=-1, keepdim=True)
        #acc_sum = 26bit(18.8)
        acc_sum = torch.floor(acc_sum * scale_factor_8)/scale_factor_8 ##소수부8
        acc_sum = torch.clip(acc_sum, -2**17, 2**17- 1/scale_factor_8)#정수부18 .소수부8
        #torch.save(acc_sum,'/home/user/HJH/transformers/src/MPWnormfile/acc_sum_18_8.pt') ##저장

        #mean계산
        mean = acc_sum /2**8 #Q(18.8) -> Q(10.8)
        mean = torch.floor(mean * scale_factor_8)/scale_factor_8 #소수부8로 precision 맞춤
        mean = mean *0.33203125 #Q(10.8)*0.8 = 10.16

        #mean = 16bit(8.8)로 만들기 = saturation
        mean = torch.clip(mean, -2**7, 2**7 - 1/scale_factor_8) #정수부8.8
        mean = torch.floor(mean * scale_factor_8)/scale_factor_8 #소수부8
        # self._save_fxp_tensor(mean, 'mean')

        #분산계산
        #X^2 = 32bit(16.16) (8.8의 제곱)
        x2 = input_fx16*input_fx16
        #X^2 = 24bit(16.8)

        x2 = torch.floor(x2 * scale_factor_8)/scale_factor_8 #소수부8 #rescaling and round
        #[try]#x2 = torch.clip(x2, -2**15, 2**15- 1/scale_factor_8)# 정수부16.8

        #acc_sum_x2= 34bit(26.8 )
        acc_sum_x2 = torch.sum(x2, dim=-1, keepdim=True)
        acc_sum_x2 = torch.floor(acc_sum_x2 * scale_factor_8)/scale_factor_8 #소수부8
        #[try]#acc_sum_x2 = torch.clip(acc_sum_x2, -2**25, 2**25- 1/scale_factor_8)# 정수부26.8
        #torch.save(acc_sum_x2,'/home/user/HJH/transformers/src/MPWnormfile/acc_sum_x2_26_8.pt') ##저장


        #mean_x2
        mean_x2 = acc_sum_x2 /2**8 #/dmodel (26.8)>>8 = 18.8
        mean_x2 = torch.floor(mean_x2 * scale_factor_8)/scale_factor_8 #소수부8
        mean_x2 = mean_x2 *0.33203125 #18.8*0.8 = 18.16

        ###mean_x2 = 26bit(16.10)
        ###mean_x2 = torch.floor(mean_x2 * scale_factor_10)/scale_factor_10 #소수부10

        #mean_x2 = 32bit(16.16)
        mean_x2 = torch.floor(mean_x2 * scale_factor_16)/scale_factor_16 #소수부16
        mean_x2 = torch.clip(mean_x2, -2**15, 2**15 - 1/scale_factor_16) #정수부16.16
        #torch.save(mean_x2,'/home/user/HJH/transformers/src/MPWnormfile/mean_x2_16_16.pt') ##저장

        #E(x)^2 = 32bit(16.16)
        temp = mean*mean
        temp = torch.floor(temp * scale_factor_16)/scale_factor_16 #소수부16
        temp = torch.clip(temp, -2**15, 2**15 - 1/scale_factor_16) #정수부16.16

        ###E(x)^2 = 32bit(16.16)  -> 26bit(16.10)로 만들기
        ###temp = torch.floor(temp * scale_factor_10)/scale_factor_10 #소수부10
        ###var = 26bit(16.10) - 26bit(16.10)


        #var = 32bit - 32bit = 32bit(16.16)  -> (8.16)로 saturation
        var = mean_x2 - temp
        var = torch.floor(var * scale_factor_16)/scale_factor_16 #소수부16
        var = torch.clip(var, -2**7, 2**7 - 1/scale_factor_16) #정수부 (8.16)
        #torch.save(var,'/home/user/HJH/transformers/src/MPWnormfile/var_8_16.pt') ##저장

        eps = self.eps #(.16)
        eps = round(eps * scale_factor_16)/scale_factor_16 #소수부16 = eps=0.0이됨
        eps = 0.0000152587890625

        v = var + eps
        # invsqrt = 1/ torch.sqrt(var + eps)
        invsqrt = self.invsqrt(v) #custom_invsqrt.py

        #inverse square root = 8.8
        invsqrt = torch.floor(invsqrt * scale_factor_8)/scale_factor_8
        invsqrt = torch.clip(invsqrt, -2**7, 2**7 - 1/scale_factor_8)
        # self._save_fxp_tensor(invsqrt, 'invsqrt')

        # 8.8  * 8.8 = 16.16
        normalized = (input_fx16 - mean) * invsqrt
        #normalized = 8.8
        normalized = torch.floor(normalized * scale_factor_8)/scale_factor_8 #소수부8
        normalized = torch.clip(normalized, -2**7, 2**7 - 1/scale_factor_8) #정수부8
        # self._save_fxp_tensor(normalized, 'normalized')

        if self.weight is not None:
            #self.weight = 8.8
            weight = torch.floor(self.weight * scale_factor_8)/scale_factor_8 #소수부8
            weight = torch.clip(weight, -2**7, 2**7 - 1/scale_factor_8) #정수부8

            #torch.save(weight,'/home/user/HJH/transformers/src/MPWnormfile/weight_8_8.pt') ##저장  
            out = normalized * weight
        
        else : out = normalized
        #8.8
        out = torch.floor(out * scale_factor_8)/scale_factor_8 #소수부8
        out = torch.clip(out, -2**7, 2**7 - 1/scale_factor_8) #정수부8
        #torch.save(out,'/home/user/HJH/transformers/src/MPWnormfile/weightedout_8_8.pt') ##저장 


        if self.bias is not None:
            #self.bias = 8.8
            bias = torch.floor(self.bias * scale_factor_8)/scale_factor_8 #소수부8
            bias = torch.clip(bias, -2**7, 2**7 - 1/scale_factor_8) #정수부8

            #torch.save(bias,'/home/user/HJH/transformers/src/MPWnormfile/bias_8_8.pt') ##저장  
            out = out + bias

        #8.8
        out = torch.floor(out * scale_factor_8)/scale_factor_8 #소수부8
        out = torch.clip(out, -2**7, 2**7 - 1/scale_factor_8) #정수부8

        #torch.save(out,'/home/user/HJH/transformers/src/MPWnormfile/biasedout_8_8.pt') ##저장 
        #(8.8)

        return out

    def forward_dual_path(self, input: Tensor) -> Tensor:
        # scale_factor_8 = 2**8
        # scale_factor_10 = 2**10
        # scale_factor_16 = 2**16

        # #input = 8.8
        # input = torch.floor(input * scale_factor_8)/scale_factor_8 #소수부8
        # input = torch.clip(input, -2**7, 2**7 - 1/scale_factor_8) #정수부8.8
        # #torch.save(input,'/home/user/HJH/transformers/src/MPWnormfile/input_8_8.pt') ##저장

        # Algorithm implementation
        B, N, D = input.shape
        device = input.device
        D_t = D - self.D_s
        Y = torch.zeros_like(input)

        for b in range(B):
            for n in range(N):
                X = input[b, n, :]

                # Identify significant dimensions
                abs_X = torch.abs(X)
                sig_values, indices = torch.topk(abs_X, self.D_s)
                sig_mask = torch.zeros_like(X, dtype=torch.bool)
                sig_mask[indices] = True

                # Path 1: Significant dimensions
                sumX_sig = torch.sum(X[sig_mask])
                sumX2_sig = torch.sum(X[sig_mask] ** 2)

                # Path 2: Trivial dimensions
                tri_mask = ~sig_mask
                sumX_tri = torch.sum(X[tri_mask])

                # Sample every Nt-th element for trivial dimensions' square sum
                sampled_tri_indices = torch.nonzero(tri_mask).squeeze(1)[::self.N_t]
                sumX2_tri = torch.sum(X[sampled_tri_indices] ** 2) * D_t

                # Calculate mean and variance
                meanX = (sumX_sig + sumX_tri) / D
                varX = (sumX2_sig + sumX2_tri) / D - meanX ** 2

                # Normalization
                Y[b, n, :] = (X - meanX) / torch.sqrt(varX + self.eps)

        normalized = Y
        
        # #normalized = 8.8
        # normalized = torch.floor(normalized * scale_factor_8)/scale_factor_8 #소수부8
        # normalized = torch.clip(normalized, -2**7, 2**7 - 1/scale_factor_8) #정수부8
        # #torch.save(normalized,'/home/user/HJH/transformers/src/MPWnormfile/normalized_8_8.pt') ##저장    

        if self.weight is not None:
            # #self.weight = 8.8
            # weight = torch.floor(self.weight * scale_factor_8)/scale_factor_8 #소수부8
            # weight = torch.clip(weight, -2**7, 2**7 - 1/scale_factor_8) #정수부8

            #torch.save(weight,'/home/user/HJH/transformers/src/MPWnormfile/weight_8_8.pt') ##저장  
            out = normalized * self.weight

        
        #8.8
        # out = torch.floor(out * scale_factor_8)/scale_factor_8 #소수부8
        # out = torch.clip(out, -2**7, 2**7 - 1/scale_factor_8) #정수부8
        #torch.save(out,'/home/user/HJH/transformers/src/MPWnormfile/weightedout_8_8.pt') ##저장 


        if self.bias is not None:
            # #self.bias = 8.8
            # bias = torch.floor(self.bias * scale_factor_8)/scale_factor_8 #소수부8
            # bias = torch.clip(bias, -2**7, 2**7 - 1/scale_factor_8) #정수부8

            #torch.save(bias,'/home/user/HJH/transformers/src/MPWnormfile/bias_8_8.pt') ##저장  
            out = out + self.bias

        # #8.8
        # out = torch.floor(out * scale_factor_8)/scale_factor_8 #소수부8
        # out = torch.clip(out, -2**7, 2**7 - 1/scale_factor_8) #정수부8

        # torch.save(out,'/home/user/HJH/transformers/src/MPWnormfile/biasedout_8_8.pt') ##저장 
        # (8.8)
        return out



class GroupNorm(Module):
    r"""Applies Group Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. :attr:`num_channels` must be divisible by
    :attr:`num_groups`. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = nn.GroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = nn.GroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = nn.GroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)
    """

    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.empty(num_channels, **factory_kwargs))
            self.bias = Parameter(torch.empty(num_channels, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        return F.group_norm(
            input, self.num_groups, self.weight, self.bias, self.eps)

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)


# TODO: ContrastiveNorm2d
# TODO: DivisiveNorm2d
# TODO: SubtractiveNorm2d
