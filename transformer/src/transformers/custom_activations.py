""" 
custom gelu (nnlut이용하여 LUT16개 사용함) 추가
"""
# Copyright 2020 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections import OrderedDict

import torch
from packaging import version
from torch import Tensor, nn

from .utils import logging


logger = logging.get_logger(__name__)


class PytorchGELUTanh(nn.Module):
    """
    A fast C implementation of the tanh approximation of the GeLU activation function. See
    https://arxiv.org/abs/1606.08415.

    This implementation is equivalent to NewGELU and FastGELU but much faster. However, it is not an exact numerical
    match due to rounding errors.
    """

    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.12.0"):
            raise ImportError(
                f"You are using torch=={torch.__version__}, but torch>=1.12.0 is required to use "
                "PytorchGELUTanh. Please upgrade torch."
            )

    def forward(self, input: Tensor) -> Tensor:
        return nn.functional.gelu(input, approximate="tanh")


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class FastGELUActivation(nn.Module):
    """
    Applies GELU approximation that is slower than QuickGELU but more accurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1.0 + torch.tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


class QuickGELUActivation(nn.Module):
    """
    Applies GELU approximation that is fast but somewhat inaccurate. See: https://github.com/hendrycks/GELUs
    """

    def forward(self, input: Tensor) -> Tensor:
        return input * torch.sigmoid(1.702 * input)


class ClippedGELUActivation(nn.Module):
    """
    Clip the range of possible GeLU outputs between [min, max]. This is especially useful for quantization purpose, as
    it allows mapping negatives values in the GeLU spectrum. For more information on this trick, please refer to
    https://arxiv.org/abs/2004.09602.

    Gaussian Error Linear Unit. Original Implementation of the gelu activation function in Google Bert repo when
    initially created.

    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))). See https://arxiv.org/abs/1606.08415
    """

    def __init__(self, min: float, max: float):
        if min > max:
            raise ValueError(f"min should be < max (got min: {min}, max: {max})")

        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x: Tensor) -> Tensor:
        return torch.clip(gelu(x), self.min, self.max)


class AccurateGELUActivation(nn.Module):
    """
    Applies GELU approximation that is faster than default and more accurate than QuickGELU. See:
    https://github.com/hendrycks/GELUs

    Implemented along with MEGA (Moving Average Equipped Gated Attention)
    """

    def __init__(self):
        super().__init__()
        self.precomputed_constant = math.sqrt(2 / math.pi)

    def forward(self, input: Tensor) -> Tensor:
        return 0.5 * input * (1 + torch.tanh(self.precomputed_constant * (input + 0.044715 * torch.pow(input, 3))))


class MishActivation(nn.Module):
    """
    See Mish: A Self-Regularized Non-Monotonic Activation Function (Misra., https://arxiv.org/abs/1908.08681). Also
    visit the official repository for the paper: https://github.com/digantamisra98/Mish
    """

    def __init__(self):
        super().__init__()
        if version.parse(torch.__version__) < version.parse("1.9.0"):
            self.act = self._mish_python
        else:
            self.act = nn.functional.mish

    def _mish_python(self, input: Tensor) -> Tensor:
        return input * torch.tanh(nn.functional.softplus(input))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


class LinearActivation(nn.Module):
    """
    Applies the linear activation function, i.e. forwarding input directly to output.
    """

    def forward(self, input: Tensor) -> Tensor:
        return input


class LaplaceActivation(nn.Module):
    """
    Applies elementwise activation based on Laplace function, introduced in MEGA as an attention activation. See
    https://arxiv.org/abs/2209.10655

    Inspired by squared relu, but with bounded range and gradient for better stability
    """

    def forward(self, input, mu=0.707107, sigma=0.282095):
        input = (input - mu).div(sigma * math.sqrt(2.0))
        return 0.5 * (1.0 + torch.erf(input))


class ReLUSquaredActivation(nn.Module):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    def forward(self, input):
        relu_applied = nn.functional.relu(input)
        squared = torch.square(relu_applied)
        return squared


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)

class CustomGELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('d', torch.tensor([
            -8.162145614624023438e+00, -6.177558898925781250e+00, -2.374777793884277344e+00, -8.664073944091796875e-01,
            -5.209192037582397461e-01, -2.167256176471710205e-01, -2.164556086063385010e-01, 6.027936190366744995e-02, 
            3.357244729995727539e-01, 6.348598003387451172e-01, 9.772109389305114746e-01, 1.914330363273620605e+00, 
            2.452649593353271484e+00, 3.153164863586425781e+00, 9.147675514221191406e+00, 
            float('inf')
        ], dtype=torch.float32))
        self.register_buffer('s', torch.tensor([
            -3.431061655282974243e-02, -3.232172876596450806e-02, -2.795374020934104919e-03, -1.104631870985031128e-01, 
            2.723709307610988617e-02, 2.196026593446731567e-01, 5.142736434936523438e-01, 4.414444267749786377e-01, 
            6.534749865531921387e-01, 8.587062954902648926e-01, 1.021672487258911133e+00, 1.121525526046752930e+00, 
            1.066059112548828125e+00, 1.019969463348388672e+00, 1.000579476356506348e+00, 1.004278540611267090e+00
        ], dtype=torch.float32))
        self.register_buffer('t', torch.tensor([
            -2.108189165592193604e-01, -1.945853531360626221e-01, -1.218453794717788696e-02, -2.678716778755187988e-01, 
            -1.485671401023864746e-01, -4.836021363735198975e-02, 1.550253480672836304e-02, -2.617612481117248535e-04, 
            -1.304284483194351196e-02, -8.194400370121002197e-02, -1.854047477245330811e-01, -2.829822897911071777e-01, 
            -1.768011450767517090e-01, -6.375940889120101929e-02, -2.619445323944091797e-03, -3.645699471235275269e-02
        ], dtype=torch.float32))

        scale_factor_8 = 2**8
        
        d = self.d
        s= self.s
        t = self.t
        
        d = torch.floor(d * scale_factor_8)/scale_factor_8 #소수부16
        d = torch.clip(d, -2**7, 127.99609375) #정수부8
        
        # s = torch.floor(s * scale_factor_16)/scale_factor_16 #소수부16
        # s = torch.clip(s, -2**15, 2**15- 1) #정수부16
        s = torch.floor(s * scale_factor_8)/scale_factor_8 #소수부8
        s = torch.clip(s, -2**7, 127.99609375) #정수부8
        
        t = torch.floor(t * scale_factor_8)/scale_factor_8 #소수부8
        t = torch.clip(t, -2**7, 127.99609375) #정수부8


 
    def forward(self, x):
        d = self.d.to(x.device)
        s = self.s.to(x.device)
        t = self.t.to(x.device)
        x = torch.where(x < -5, torch.zeros_like(x), x)
        x = torch.where(x > 5, x, self.compute_custom_gelu(x))
        return x
    
    def compute_custom_gelu(self, x):
        idx = torch.searchsorted(self.d, x, right=True)



        #lut연산
        s_multiply_x = self.s[idx] * x  ## (8.8)*(8.8)=32bit 16.16?

        s_multiply_x = torch.floor(s_multiply_x * (2**8))/(2**8) 
        s_multiply_x = torch.clip(s_multiply_x, -2**7, 127.99609375)#-2**15,  32767.99998474121) # 16.16 format min , max

        result =  s_multiply_x + self.t[idx] ## (8.8)+8.8

        


        # result format 8.8
        result = torch.floor(result * (2**8))/(2**8) #
        result = torch.clip(result, -2**7, 127.99609375) # 8.8 format min , max
        # result = torch.floor(result * (2**15))/(2**15) #
        # result = torch.clip(result, -2**15,  32767.99998474121)
        x_transformed = torch.where(x < -5, torch.zeros_like(x), result)
       

        #original
        # x_transformed = torch.where(x < -5, torch.zeros_like(x), self.s[idx] * x + self.t[idx]) #안
        return x_transformed

ACT2CLS = {
    "gelu": GELUActivation,
    "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
    "gelu_fast": FastGELUActivation,
    "gelu_new": NewGELUActivation,
    "gelu_python": (GELUActivation, {"use_gelu_python": True}),
    "gelu_pytorch_tanh": PytorchGELUTanh,
    "gelu_accurate": AccurateGELUActivation,
    "laplace": LaplaceActivation,
    "leaky_relu": nn.LeakyReLU,
    "linear": LinearActivation,
    "mish": MishActivation,
    "quick_gelu": QuickGELUActivation,
    "relu": nn.ReLU,
    "relu2": ReLUSquaredActivation,
    "relu6": nn.ReLU6,
    "sigmoid": nn.Sigmoid,
    "silu": nn.SiLU,
    "swish": nn.SiLU,
    "tanh": nn.Tanh,
    "CustomGELU": CustomGELU,
}
ACT2FN = ClassInstantier(ACT2CLS)


def get_activation(activation_string):
    if activation_string in ACT2FN:
        return ACT2FN[activation_string]
    else:
        raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")


# For backwards compatibility with: from activations import gelu_python
gelu_python = get_activation("gelu_python")
gelu_new = get_activation("gelu_new")
gelu = get_activation("gelu")
gelu_fast = get_activation("gelu_fast")
quick_gelu = get_activation("quick_gelu")
silu = get_activation("silu")
mish = get_activation("mish")
linear_act = get_activation("linear")
CustomGELU = get_activation("CustomGELU")