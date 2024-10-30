import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.pack_utils import pack_sign, unpack_sign, unpack_sign_v2, unpack_sign_triton
from fused_forward import reconstruct_w_triton, reconstruct_and_forward_triton, unpack_and_reconstruct_w_triton, unpack_and_reconstruct_w_triton_v2, unpack_and_reconstruct_w_triton_v3
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
    def __init__(self, w_bit, in_features, out_features, k, bias, no_avd, parallel_forward, dev):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.k = k if not no_avd else (in_features * out_features) // (16*(in_features+out_features)) + k
        self.no_avd = no_avd
        self.parallel_forward = parallel_forward
        self.name = None
        for i in range(self.w_bit):
            if not no_avd:
                self.register_buffer(f'qweight_{i}', torch.zeros((out_features * in_features) // 8, dtype=torch.uint8, device=dev))
            self.register_buffer(f'u_{i}', torch.zeros(out_features, self.k, dtype=torch.float16, device=dev))
            self.register_buffer(f'vt_{i}', torch.zeros(self.k, in_features, dtype=torch.float16, device=dev))
            if bias:
                self.register_buffer(f'bias_{i}', torch.zeros(out_features, dtype=torch.float16, device=dev))

    def prepare_for_saving(self):
        # Split concatenated tensors into individual tensors for saving
        for i in range(self.w_bit):
            if not self.no_avd:
                setattr(self, f'qweight_{i}', self.qweight[i])
            setattr(self, f'u_{i}', self.u[i])
            setattr(self, f'vt_{i}', self.vt[i])
            if self.bias is not None:
                setattr(self, f'bias_{i}', self.bias[i])
        
        for key in ['qweight', 'u', 'vt', 'bias']:
            if hasattr(self, key):
                delattr(self, key)

    def stack_blocks(self):
        # Combine individual tensors into concatenated tensors after loading
        if not self.no_avd:
            self.qweight = torch.stack([getattr(self, f'qweight_{i}') for i in range(self.w_bit)])
        self.u = torch.stack([getattr(self, f'u_{i}') for i in range(self.w_bit)])
        self.vt = torch.stack([getattr(self, f'vt_{i}') for i in range(self.w_bit)])
        if hasattr(self, 'bias_0'):
            self.bias = torch.stack([getattr(self, f'bias_{i}') for i in range(self.w_bit)])
        # Remove individual tensors to save memory
        for i in range(self.w_bit):
            if not self.no_avd:
                delattr(self, f'qweight_{i}')
            delattr(self, f'u_{i}')
            delattr(self, f'vt_{i}')
            if hasattr(self, f'bias_{i}'):
                delattr(self, f'bias_{i}')
    
    def decompose_weight(self, original_weight):
        weight = original_weight.to(torch.float32)
        weight_shape = weight.data.shape
        singular_values = []
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
    def from_linear(cls, linear, w_bit, k=1, bias=True, no_avd=False, parallel_forward=False, name=None, init_only=False):
        # assuming not bias for original Linear layer here since most of the models don't use bias these days
        assert linear.bias is None
        qlinear = cls(w_bit=w_bit, in_features=linear.in_features, out_features=linear.out_features, bias=bias, k=k, no_avd=no_avd, parallel_forward=parallel_forward, dev=linear.weight.device)
        qlinear.name = name
        if init_only:
            return qlinear
        qlinear.decompose_weight(linear.weight)
        return qlinear

    @torch.no_grad()
    def forward(self, x):
        if self.parallel_forward:
            return self.fast_forward(x)
        if hasattr(self, 'w') and self.w is not None:
            w = self.w
        else:
            w = torch.zeros((self.out_features, self.in_features), dtype=torch.float16, device=x.device)
            for i in range(self.w_bit):
                if self.no_avd:
                    w += compose_weight(None, getattr(self, f'u_{i}'), getattr(self, f'vt_{i}'), self.no_avd, w.shape)
                else:
                    w += compose_weight(getattr(self, f'qweight_{i}'), getattr(self, f'u_{i}'), getattr(self, f'vt_{i}'), self.no_avd, w.shape)
        result = x @ w.T

        return result
    
    @torch.no_grad()
    def fast_forward(self, x):
        assert not self.no_avd

        # unpacked_sign = unpack_sign_triton(self.qweight.view(-1), torch.Size([self.w_bit, self.out_features, self.in_features]))

        # w = torch.bmm(self.u, self.vt)
        # w = torch.matmul(self.u, self.vt)
        # w = (w.mul_(unpacked_sign)).sum(dim=0).contiguous()
        # w = (self.u @ self.vt).mul_(unpacked_sign).sum(dim=0).contiguous()
        # w.mul_(unpacked_sign.to(torch.float16))
        # w = w.sum(dim=0)

        # w[unpacked_sign==-1] = -w[unpacked_sign==-1]
        # w = w.sum(dim=0)

        # w_abs = torch.matmul(self.u, self.vt)
        # w = torch.where(unpacked_sign==-1, -w_abs, w_abs)
        # w = w.sum(dim=0)
        # return F.linear(x, (self.u @ self.vt).mul_(unpacked_sign).sum(dim=0).contiguous(), bias=None)

        # w = reconstruct_w_triton(unpacked_sign, self.u, self.vt)
        # return F.linear(x, w, bias=None)

        # result = reconstruct_and_forward_triton(unpacked_sign, self.u, self.vt, x)
        # return result

        # w = unpack_and_reconstruct_w_triton(self.qweight, self.u, self.vt)
        # return F.linear(x, w, bias=None)

        w = unpack_and_reconstruct_w_triton_v2(self.qweight, self.u, self.vt)
        return F.linear(x, w, bias=None)
        
        # w = unpack_and_reconstruct_w_triton_v3(self.qweight, self.u, self.vt)
        # return F.linear(x, w, bias=None)

    def memory_per_bit(self):
        memory = 0
        if not self.no_avd:
            memory += self.qweight_0.numel() * self.qweight_0.element_size()
        for module in [self.u_0, self.vt_0]:
            memory += module.numel() * module.element_size()
        return memory

    def pop_one_bit(self):
        # here we don't really discard the weights
        self.w_bit -= 1
    
    def add_one_bit(self):
        self.w_bit += 1
    
    def set_bit(self, bit):
        self.w_bit = bit
