import torch
import torch.nn as nn
from bitstack.utils.pack_utils import pack_sign, unpack_sign

def compose_weight(weight_sign, u, vt, no_avd=False, weight_shape=None, packed_sign=True):
    if no_avd:
        return u @ vt
    else:
        dtype = u.dtype
        if packed_sign:
            weight_sign = unpack_sign(weight_sign, weight_shape)
        w = weight_sign.to(dtype) * torch.matmul(u, vt)
        return w


class ScaledActivation(nn.Module):
    def __init__(self, module, scales):
        super().__init__()
        self.act = module
        self.scales = nn.Parameter(scales.data)

    def forward(self, x):
        return self.act(x) / self.scales.view(1, 1, -1).to(x.device)

class BitStackLinear(nn.Module):
    def __init__(self, w_bit, in_features, out_features, k, bias, no_avd, dev):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.k = k if not no_avd else (in_features * out_features) // (16*(in_features+out_features)) + k
        self.no_avd = no_avd

        for i in range(self.w_bit):
            if not no_avd:
                self.register_buffer(f'qweight_{i}', torch.zeros((out_features * in_features) // 8, dtype=torch.uint8, device=dev))
            self.register_buffer(f'u_{i}', torch.zeros(out_features, self.k, dtype=torch.float16, device=dev))
            self.register_buffer(f'vt_{i}', torch.zeros(self.k, in_features, dtype=torch.float16, device=dev))
            if bias:
                self.register_buffer(f'bias_{i}', torch.zeros(out_features, dtype=torch.float16, device=dev))
            else:
                self.register_buffer(f'bias_{i}', None)

        self.name = None
    
    def decompose_weight(self, original_weight):
        weight = original_weight.to(torch.float32)
        weight_shape = weight.data.shape
        for i in range(self.w_bit):
            weight_sign = weight.sign()
            weight_sign[weight_sign == 0] = 1
            weight_abs = weight.abs()
            if self.no_avd:
                u, s, vt = torch.linalg.svd(weight, full_matrices=False)
            else:
                u, s, vt = torch.linalg.svd(weight_abs, full_matrices=False)
            
            sk = s[:self.k]
            sqrt_sk = torch.diag(torch.sqrt(sk))
            us_k = u[:, :self.k] @ sqrt_sk
            vt_k = sqrt_sk @ vt[:self.k, :]
            packed_weight_sign = None
            if not self.no_avd:
                packed_weight_sign = pack_sign(weight_sign.to(torch.int8))
                assert unpack_sign(packed_weight_sign, weight_shape).equal(weight_sign.to(torch.int8))
                setattr(self, f'qweight_{i}', packed_weight_sign)
            setattr(self, f'u_{i}', us_k.to(torch.float16).contiguous())
            setattr(self, f'vt_{i}', vt_k.to(torch.float16).contiguous())
            w_i = compose_weight(packed_weight_sign, us_k, vt_k, self.no_avd, weight_shape)
            assert torch.isinf(w_i).sum() == 0
            assert torch.isnan(w_i).sum() == 0
            weight = weight - w_i

    @classmethod
    def from_linear(cls, linear, w_bit, k=1, bias=True, no_avd=False, name=None, init_only=False):
        # assuming not bias for original Linear layer here since most of the models don't use bias these days
        assert linear.bias is None
        qlinear = cls(w_bit=w_bit, in_features=linear.in_features, out_features=linear.out_features, bias=bias, k=k, no_avd=no_avd, dev=linear.weight.device)
        qlinear.name = name
        if init_only:
            return qlinear
        qlinear.decompose_weight(linear.weight)
        return qlinear

    @torch.no_grad()
    def forward(self, x):
        w = torch.zeros((self.out_features, self.in_features), dtype=torch.float16, device=x.device)
        for i in range(self.w_bit):
            if self.no_avd:
                w += compose_weight(None, getattr(self, f'u_{i}'), getattr(self, f'vt_{i}'), self.no_avd, w.shape)
            else:
                w += compose_weight(getattr(self, f'qweight_{i}'), getattr(self, f'u_{i}'), getattr(self, f'vt_{i}'), self.no_avd, w.shape)
        result = x @ w.T
        return result
    
    def memory_per_bit(self):
        memory = 0
        if not self.no_avd:
            memory += self.qweight_0.numel() * self.qweight_0.element_size()
        for module in [self.u_0, self.vt_0]:
            memory += module.numel() * module.element_size()
        return memory

    def pop_one_bit(self):
        # here we don't really discard the weights, only for quick evaluation
        self.w_bit -= 1
    
    def add_one_bit(self):
        self.w_bit += 1
    
    def set_bit(self, bit):
        self.w_bit = bit