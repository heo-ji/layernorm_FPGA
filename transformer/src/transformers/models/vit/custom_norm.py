"""
torch.nn.modules.normalization에서 복붙
class Custom_LayerNorm 
"""

from .custom_invsqrt import custom_invsqrt #추가함


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
#         input_fx16 = torch.clip(input_fx16, -2**7, 127.99609375) #정수부8.8
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
#         mean = torch.clip(mean, -2**7, 127.99609375) #정수부8.8
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
#         mean_x2 = torch.clip(mean_x2, -2**15, 32767.99998474121) #정수부16.16    
#         #torch.save(mean_x2,'/home/user/HJH/transformers/src/MPWnormfile/mean_x2_16_16.pt') ##저장 

#         #E(x)^2 = 32bit(16.16)
#         temp = mean*mean
#         temp = torch.floor(temp * scale_factor_16)/scale_factor_16 #소수부16

#         ###E(x)^2 = 32bit(16.16)  -> 26bit(16.10)로 만들기 
#         ###temp = torch.floor(temp * scale_factor_10)/scale_factor_10 #소수부10
#         temp = torch.clip(temp, -2**15, 32767.99998474121) #정수부16.16
#         ###var = 26bit(16.10) - 26bit(16.10)


#         #var = 32bit - 32bit = 32bit(16.16)  -> (8.16)
#         var = mean_x2 - temp

#         var = torch.floor(var * scale_factor_16)/scale_factor_16 #소수부16
#         var = torch.clip(var, -2**7, 127.99998474121094) #정수부 (8.16)
#         #torch.save(var,'/home/user/HJH/transformers/src/MPWnormfile/var_8_16.pt') ##저장 

#         eps = self.eps #(.16)
#         eps = round(eps * scale_factor_16)/scale_factor_16 #소수부16 = eps=0.0이됨
#         eps = 0.0000152587890625

#         v=var + eps
#         # invsqrt = 1/ torch.sqrt(var + eps)
#         invsqrt = self.invsqrt(v) #custom_invsqrt.py

#         #inverse square root = 8.8
#         invsqrt = torch.floor(invsqrt * scale_factor_8)/scale_factor_8
#         invsqrt = torch.clip(invsqrt, -2**7, 127.99609375)
#         #torch.save(invsqrt,'/home/user/HJH/transformers/src/MPWnormfile/invsqrt_8_8.pt') ##저장 

#         # 8.8  * 8.8 = 16.16
#         normalized = (input_fx16 - mean) * invsqrt 
#         #normalized = 8.8
#         normalized = torch.floor(normalized * scale_factor_8)/scale_factor_8 #소수부8
#         normalized = torch.clip(normalized, -2**7, 127.99609375) #정수부8
#         #torch.save(normalized,'/home/user/HJH/transformers/src/MPWnormfile/normalized_8_8.pt') ##저장    

#         if self.weight is not None:
#             #self.weight = 8.8
#             weight = torch.floor(self.weight * scale_factor_8)/scale_factor_8 #소수부8
#             weight = torch.clip(weight, -2**7, 127.99609375) #정수부8

#             #torch.save(weight,'/home/user/HJH/transformers/src/MPWnormfile/weight_8_8.pt') ##저장  
#             out = normalized * weight
        
#         else : out = normalized
#         #8.8
#         out = torch.floor(out * scale_factor_8)/scale_factor_8 #소수부8
#         out = torch.clip(out, -2**7, 127.99609375) #정수부8
#         #torch.save(out,'/home/user/HJH/transformers/src/MPWnormfile/weightedout_8_8.pt') ##저장 


#         if self.bias is not None:
#             #self.bias = 8.8
#             bias = torch.floor(self.bias * scale_factor_8)/scale_factor_8 #소수부8
#             bias = torch.clip(bias, -2**7, 127.99609375) #정수부8

#             #torch.save(bias,'/home/user/HJH/transformers/src/MPWnormfile/bias_8_8.pt') ##저장  
#             out = out + bias

#         #8.8
#         out = torch.floor(out * scale_factor_8)/scale_factor_8 #소수부8
#         out = torch.clip(out, -2**7, 127.99609375) #정수부8

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

    def __init__(self, normalized_shape: _shape_t, eps: float = 1e-5, method: str = 'original', elementwise_affine: bool = True,
                 bias: bool = True, device=None, dtype=None) -> None:
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
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    def forward_original(self, input: Tensor) -> Tensor:
        return F.layer_norm(input, self.normalized_shape, self.weight, self.bias, self.eps)
    
    def forward_fxp88(self, input: Tensor) -> Tensor:
        dmodel = input.size(2)
        
        scale_factor_8 = 2**8
        scale_factor_10 = 2**10
        scale_factor_16 = 2**16

        #input = 8.8
        input_fx16 = torch.floor(input * scale_factor_8)/scale_factor_8 #소수부8
        input_fx16 = torch.clip(input_fx16, -2**7, 127.99609375) #정수부8.8
        #torch.save(input_fx16,'/home/user/HJH/transformers/src/MPWnormfile/input_8_8.pt') ##저장

        acc_sum = torch.sum(input_fx16, dim=-1, keepdim=True)
        #acc_sum = 26bit(18.8)
        acc_sum = torch.floor(acc_sum * scale_factor_8)/scale_factor_8 #??필요한가? #소수부8
        acc_sum = torch.clip(acc_sum, -2**17, 2**17- 1)#??필요한가? 정수부18
        #torch.save(acc_sum,'/home/user/HJH/transformers/src/MPWnormfile/acc_sum_18_8.pt') ##저장

        #mean계산
        mean = acc_sum /2**8 #/dmodel
        mean = torch.floor(mean * scale_factor_8)/scale_factor_8 #소수부8
        mean = mean *0.33203125 #18.8*0.8 = 18.16

        #mean = 16bit(8.8)로 만들기
        mean = torch.clip(mean, -2**7, 127.99609375) #정수부8.8
        mean = torch.floor(mean * scale_factor_8)/scale_factor_8 #소수부8
        #torch.save(mean,'/home/user/HJH/transformers/src/MPWnormfile/mean_8_8.pt') ##저장

        #분산계산
        #X^2 = 32bit(16.16) (8.8의 제곱)
        x2 = input_fx16*input_fx16
        #X^2 = 24bit(16.8)
        # x2 = torch.clip(x2, -2**15, 2**15- 1)#??필요한가? 정수부16
        x2 = torch.floor(x2 * scale_factor_8)/scale_factor_8 #소수부8 #rescaling and round
        
        #acc_sum_x2= 34bit(26.8 )
        acc_sum_x2 = torch.sum(x2, dim=-1, keepdim=True)
        acc_sum_x2 = torch.floor(acc_sum_x2 * scale_factor_8)/scale_factor_8 #소수부8
        #torch.save(acc_sum_x2,'/home/user/HJH/transformers/src/MPWnormfile/acc_sum_x2_26_8.pt') ##저장
        
        
        #mean_x2
        mean_x2 = acc_sum_x2 /2**8 #/dmodel (26.8)>>8 = 16.8
        mean_x2 = torch.floor(mean_x2 * scale_factor_8)/scale_factor_8 #소수부8
        mean_x2 = mean_x2 *0.33203125 #18.8*0.8 = 18.16

        ###mean_x2 = 26bit(16.10)                                 
        ###mean_x2 = torch.floor(mean_x2 * scale_factor_10)/scale_factor_10 #소수부10

        #mean_x2 = 32bit(16.16)   
        mean_x2 = torch.floor(mean_x2 * scale_factor_16)/scale_factor_16 #소수부16
        mean_x2 = torch.clip(mean_x2, -2**15, 32767.99998474121) #정수부16.16    
        #torch.save(mean_x2,'/home/user/HJH/transformers/src/MPWnormfile/mean_x2_16_16.pt') ##저장 

        #E(x)^2 = 32bit(16.16)
        temp = mean*mean
        temp = torch.floor(temp * scale_factor_16)/scale_factor_16 #소수부16

        ###E(x)^2 = 32bit(16.16)  -> 26bit(16.10)로 만들기 
        ###temp = torch.floor(temp * scale_factor_10)/scale_factor_10 #소수부10
        temp = torch.clip(temp, -2**15, 32767.99998474121) #정수부16.16
        ###var = 26bit(16.10) - 26bit(16.10)


        #var = 32bit - 32bit = 32bit(16.16)  -> (8.16)
        var = mean_x2 - temp

        var = torch.floor(var * scale_factor_16)/scale_factor_16 #소수부16
        var = torch.clip(var, -2**7, 127.99998474121094) #정수부 (8.16)
        #torch.save(var,'/home/user/HJH/transformers/src/MPWnormfile/var_8_16.pt') ##저장 

        eps = self.eps #(.16)
        eps = round(eps * scale_factor_16)/scale_factor_16 #소수부16 = eps=0.0이됨
        eps = 0.0000152587890625

        v = var + eps
        # invsqrt = 1/ torch.sqrt(var + eps)
        invsqrt = self.invsqrt(v) #custom_invsqrt.py

        #inverse square root = 8.8
        invsqrt = torch.floor(invsqrt * scale_factor_8)/scale_factor_8
        invsqrt = torch.clip(invsqrt, -2**7, 127.99609375)
        #torch.save(invsqrt,'/home/user/HJH/transformers/src/MPWnormfile/invsqrt_8_8.pt') ##저장 

        # 8.8  * 8.8 = 16.16
        normalized = (input_fx16 - mean) * invsqrt 
        #normalized = 8.8
        normalized = torch.floor(normalized * scale_factor_8)/scale_factor_8 #소수부8
        normalized = torch.clip(normalized, -2**7, 127.99609375) #정수부8
        #torch.save(normalized,'/home/user/HJH/transformers/src/MPWnormfile/normalized_8_8.pt') ##저장    

        if self.weight is not None:
            #self.weight = 8.8
            weight = torch.floor(self.weight * scale_factor_8)/scale_factor_8 #소수부8
            weight = torch.clip(weight, -2**7, 127.99609375) #정수부8

            #torch.save(weight,'/home/user/HJH/transformers/src/MPWnormfile/weight_8_8.pt') ##저장  
            out = normalized * weight
        
        else : out = normalized
        #8.8
        out = torch.floor(out * scale_factor_8)/scale_factor_8 #소수부8
        out = torch.clip(out, -2**7, 127.99609375) #정수부8
        #torch.save(out,'/home/user/HJH/transformers/src/MPWnormfile/weightedout_8_8.pt') ##저장 


        if self.bias is not None:
            #self.bias = 8.8
            bias = torch.floor(self.bias * scale_factor_8)/scale_factor_8 #소수부8
            bias = torch.clip(bias, -2**7, 127.99609375) #정수부8

            #torch.save(bias,'/home/user/HJH/transformers/src/MPWnormfile/bias_8_8.pt') ##저장  
            out = out + bias

        #8.8
        out = torch.floor(out * scale_factor_8)/scale_factor_8 #소수부8
        out = torch.clip(out, -2**7, 127.99609375) #정수부8

        #torch.save(out,'/home/user/HJH/transformers/src/MPWnormfile/biasedout_8_8.pt') ##저장 
        #(8.8)
        return out

    def forward_dual_path(self, input: Tensor) -> Tensor:
        self.D_s = 384  # example value for the number of significant dimensions
        self.N_t = 5  # sampling rate for trivial dimensions
        # scale_factor_8 = 2**8
        # scale_factor_10 = 2**10
        # scale_factor_16 = 2**16

        # #input = 8.8
        # input = torch.floor(input * scale_factor_8)/scale_factor_8 #소수부8
        # input = torch.clip(input, -2**7, 127.99609375) #정수부8.8
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
        # normalized = torch.clip(normalized, -2**7, 127.99609375) #정수부8
        # #torch.save(normalized,'/home/user/HJH/transformers/src/MPWnormfile/normalized_8_8.pt') ##저장    

        if self.weight is not None:
            # #self.weight = 8.8
            # weight = torch.floor(self.weight * scale_factor_8)/scale_factor_8 #소수부8
            # weight = torch.clip(weight, -2**7, 127.99609375) #정수부8

            #torch.save(weight,'/home/user/HJH/transformers/src/MPWnormfile/weight_8_8.pt') ##저장  
            out = normalized * self.weight

        
        #8.8
        # out = torch.floor(out * scale_factor_8)/scale_factor_8 #소수부8
        # out = torch.clip(out, -2**7, 127.99609375) #정수부8
        #torch.save(out,'/home/user/HJH/transformers/src/MPWnormfile/weightedout_8_8.pt') ##저장 


        if self.bias is not None:
            # #self.bias = 8.8
            # bias = torch.floor(self.bias * scale_factor_8)/scale_factor_8 #소수부8
            # bias = torch.clip(bias, -2**7, 127.99609375) #정수부8

            #torch.save(bias,'/home/user/HJH/transformers/src/MPWnormfile/bias_8_8.pt') ##저장  
            out = out + self.bias

        # #8.8
        # out = torch.floor(out * scale_factor_8)/scale_factor_8 #소수부8
        # out = torch.clip(out, -2**7, 127.99609375) #정수부8

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
