import torch
from torch import nn

class SoftmaxFixedPoint:
    def __init__(self, method: str = 'original', frac: int = 8):
        self.method = method
        self.frac = frac
    
    def softmax(self, input_tensor: torch.Tensor) -> torch.Tensor:
        if self.method == 'cordic':
            return self.softmax_cordic(input_tensor)
        elif self.method == 'original':
            return nn.functional.softmax(input_tensor, dim=-1)
        elif self.method == 'base2':
            return self.custom_log2e(input_tensor)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def softmax_cordic(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input = (input_tensor * 256)
        input = input.type(torch.int64)
        exp = self.exp_large(input)

        # exp_sum = torch.sum(exp, dim=-1, keepdim=True)
        # out = exp / exp_sum


        fpexp = exp.type(torch.float32)/256
        exp_sum = torch.sum(fpexp, dim=-1, keepdim=True)
        out = fpexp / exp_sum
        return out

    
    def custom_log2e(self, x) :
        scale_factor_8 = 2**8
        #8.8
        input = torch.floor(x * scale_factor_8)/scale_factor_8
        input = torch.clip(input, -2**7, 2**7 - 1/scale_factor_8)

        

        max_input = torch.max(input)
        input = input - max_input
        
        out = torch.zeros(x.shape).to(x.device)
        log2e_x = 2 ** (x.mul(1.4375).round())
        sum_log2e_x = torch.sum(log2e_x, dim=-1, keepdim=True)
        out = log2e_x / sum_log2e_x

        #8.8
        out = torch.floor(out * scale_factor_8)/scale_factor_8
        out = torch.clip(out, -2**7, 2**7 - 1/scale_factor_8)
        
        return out
    
    def exp_large(self, input_tensor: torch.Tensor) -> torch.Tensor:
        ln_2 = 177
        rad_to_deg = 14668
        is_neg = input_tensor < 0
        is_pos = input_tensor > 0
        input_tensor[is_neg] = -input_tensor[is_neg]
        r = (input_tensor % ln_2).type(torch.int64)
        q = (input_tensor / ln_2)
        r_deg = ((r * rad_to_deg) / (2 ** 8)).type(torch.int64)
        r_deg[is_neg] = -r_deg[is_neg]

        exp_large = torch.zeros_like(input_tensor).type(torch.int64)
        exp = self.hyperbolic_cordic_16(r_deg)

        exp_shift_right = torch.div(exp, (2 ** torch.floor(q))).type(torch.int64)
        exp_shift_left = torch.mul(exp, (2 ** torch.floor(q))).type(torch.int64)

        exp_large[is_neg] = exp_shift_right[is_neg]
        exp_large[is_pos] = exp_shift_left[is_pos]

        return exp_large

    def hyperbolic_cordic_iteration(self, iter, x, y, z, angle):
        great_equal = (z >= 0).to(x.device)
        less = (z < 0).to(x.device)
        new_x = torch.zeros_like(x).to(x.device)
        new_y = torch.zeros_like(y).to(x.device)
        new_z = torch.zeros_like(z).to(x.device)

        x_shifted = torch.div(x, (2**iter), rounding_mode='floor')
        y_shifted = torch.div(y, (2**iter), rounding_mode='floor')

        new_x[great_equal] = x[great_equal] + y_shifted[great_equal]
        new_x[less] = x[less] - y_shifted[less]

        new_y[great_equal] = y[great_equal] + x_shifted[great_equal]
        new_y[less] = y[less] - x_shifted[less]

        new_z[great_equal] = z[great_equal] - angle
        new_z[less] = z[less] + angle

        return new_x, new_y, new_z

    def hyperbolic_cordic_16(self, input):
        arc_tan = [8057, 3746, 1843, 918, 459, 229, 115, 57, 29, 14, 7, 4, 2, 1, 0]
        x = torch.zeros_like(input).to(input.device) + 309
        y = torch.zeros_like(input).to(input.device)
        z = input.to(input.device)

        for i in range(1, 16):
            x, y, z = self.hyperbolic_cordic_iteration(i, x, y, z, arc_tan[i - 1])
            if i == 4 or i == 13:
                x, y, z = self.hyperbolic_cordic_iteration(i, x, y, z, arc_tan[i - 1])

        return x + y