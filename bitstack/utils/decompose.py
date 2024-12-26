import torch
import gc
from tqdm import tqdm

from bitstack.modules.BitStackLinear import BitStackLinear
from bitstack.utils.scale_utils import get_blocks, get_named_linears, set_op_by_name
@torch.no_grad()
def decompose(model, wbits, k, signed, fused_level=0, init_only=False, compression_config=None):
    layers = get_blocks(model)
    for i in tqdm(range(len(layers)), desc="Decomposing"):
        layer = layers[i]
        named_linears = get_named_linears(layer)
        for name, module in named_linears.items():
            layer_name = 'layer'+'.'+str(i)+'.'+name
            if init_only:
                if compression_config is not None:
                    load_wbits = compression_config[layer_name]
                else:
                    load_wbits = wbits
                qlinear = BitStackLinear.from_linear(module, w_bit=load_wbits, k=k, bias=module.bias is not None, no_avd=signed, fused_level=fused_level, name=layer_name, init_only=init_only)
                set_op_by_name(layer, name, qlinear)
            else:
                module.cuda()
                qlinear = BitStackLinear.from_linear(module, w_bit=wbits, k=k, bias=module.bias is not None, no_avd=signed, fused_level=fused_level, name=layer_name, init_only=init_only)
                module.cpu()
                qlinear.cpu()
                set_op_by_name(layer, name, qlinear)
                torch.cuda.empty_cache()
                gc.collect()
    torch.cuda.empty_cache()
    gc.collect()