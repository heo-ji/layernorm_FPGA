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


#input value is 26bit(16.10)
#eps = (0.16)(10'b0000_0000_0000_0001)

class custom_invsqrt(Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('d', torch.tensor([
            6.268789293244481087e-04,
            6.272371974773705006e-04,
            6.339621031656861305e-04,
            7.214249926619231701e-04,
            7.731573423370718956e-04,
            8.954962831921875477e-04,
            1.016209367662668228e-03,
            1.103247166611254215e-03,
            1.877601258456707001e-03,
            2.165360609069466591e-03,
            2.225812291726469994e-03,
            2.909098984673619270e-03,
            3.036817768588662148e-03,
            3.424482187256217003e-03,
            4.042495042085647583e-03,
            4.889855161309242249e-02,
            1.280726194381713867e-01,
            2.993504405021667480e-01,
            5.197208523750305176e-01,
            7.865777611732482910e-01,
            7.866426110267639160e-01,
            7.869081497192382812e-01,
            1.161421656608581543e+00,
            1.792997837066650391e+00,
            1.795408725738525391e+00,
            3.034406661987304688e+00,
            3.036806821823120117e+00,
            7.343480587005615234e+00,
            7.348295688629150391e+00,
            3.036649703979492188e+01,
            3.415852737426757812e+01,
            float('inf')
        ], dtype=torch.float32))
        self.register_buffer('s', torch.tensor([
            -4.376399609375000000e+04,
            -3.907193750000000000e+04,
            -3.446287890625000000e+04,
            -2.972374609375000000e+04,
            -2.505638867187500000e+04,
            -2.096733398437500000e+04,
            -1.699769335937500000e+04,
            -1.441686718750000000e+04,
            -1.208231445312500000e+04,
            -1.009773242187500000e+04,
            -8.188501953125000000e+03,
            -6.290132324218750000e+03,
            -4.649764648437500000e+03,
            -3.025012207031250000e+03,
            -1.526271972656250000e+03,
            -1.365869903564453125e+02,
            -2.016152000427246094e+01,
            -4.814074516296386719e+00,
            -1.902751922607421875e+00,
            -9.313918948173522949e-01,
            -1.224234580993652344e+00,
            -1.550122499465942383e+00,
            -5.336802005767822266e-01,
            -2.878744602203369141e-01,
            -3.544176220893859863e-01,
            -1.415169537067413330e-01,
            -4.163529276847839355e-01,
            -5.445432662963867188e-02,
            -2.348640263080596924e-01,
            -7.050544023513793945e-03,
            -4.228321462869644165e-02,
            0.000000000000000000e+00

        ], dtype=torch.float32))
        self.register_buffer('t', torch.tensor([
            6.754576873779296875e+01,
            6.460441589355468750e+01,
            6.171343612670898438e+01,
            5.870901107788085938e+01,
            5.534186172485351562e+01,
            5.218037796020507812e+01,
            4.862558364868164062e+01,
            4.600291824340820312e+01,
            4.342733383178710938e+01,
            3.970107650756835938e+01,
            3.556690597534179688e+01,
            3.134149932861328125e+01,
            2.656950759887695312e+01,
            2.163543319702148438e+01,
            1.650302124023437500e+01,
            1.088522529602050781e+01,
            5.192188262939453125e+00,
            3.226601123809814453e+00,
            2.355095148086547852e+00,
            1.850259065628051758e+00,
            2.080602645874023438e+00,
            2.336960077285766602e+00,
            1.537113308906555176e+00,
            1.251629114151000977e+00,
            1.370940923690795898e+00,
            9.886971712112426758e-01,
            1.822661280632019043e+00,
            7.236450910568237305e-01,
            2.048480033874511719e+00,
            3.744393587112426758e-01,
            1.444332242012023926e+00,
            0.000000000000000000e+00

        ], dtype=torch.float32))

        d = self.d
        s= self.s
        t = self.t
 
        """ 
        d = pd.read_csv('/home/user/HJH/transformers/src/transformers/models/bert/invsqrt_csv_file/inv_SQRT_d.csv')
        s = pd.read_csv('/home/user/HJH/transformers/src/transformers/models/bert/invsqrt_csv_file/inv_SQRT_s.csv')
        t = pd.read_csv('/home/user/HJH/transformers/src/transformers/models/bert/invsqrt_csv_file/inv_SQRT_t.csv')
        
        d = torch.tensor(d.values, dtype=torch.float32)
        s = torch.tensor(s.values, dtype=torch.float32)
        t = torch.tensor(t.values, dtype=torch.float32) """
        
        scale_factor_16 = 2**16
        scale_factor_8 = 2**8
        
        d = torch.floor(d * scale_factor_16)/scale_factor_16 #소수부16
        # d = torch.clip(d, -2**15, 2**15- 1) #정수부16
        d = torch.clip(d, -2**7, 127.99998474121094) #정수부8
        
        # s = torch.floor(s * scale_factor_16)/scale_factor_16 #소수부16
        # s = torch.clip(s, -2**15, 2**15- 1) #정수부16
        s = torch.floor(s * scale_factor_8)/scale_factor_8 #소수부8
        s = torch.clip(s, -2**7, 127.99609375) #정수부8
        
        t = torch.floor(t * scale_factor_8)/scale_factor_8 #소수부8
        t = torch.clip(t, -2**7, 127.99609375) #정수부8

        """ d=d.squeeze()
        s=s.squeeze()
        t=t.squeeze()
        self.register_buffer('d', torch.tensor(d, dtype=torch.float32))
        self.register_buffer('s', torch.tensor(s, dtype=torch.float32))
        self.register_buffer('t', torch.tensor(t, dtype=torch.float32)) """



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
        s_multiply_x = torch.clip(s_multiply_x, -2**7,  127.99609375) #정수부8
        
        
        result =  s_multiply_x + self.t[idx_clamped] ## (8.8)+8.8
        return result




