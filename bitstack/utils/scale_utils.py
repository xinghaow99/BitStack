import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer, MistralRMSNorm, MistralForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2DecoderLayer, Qwen2ForCausalLM
from collections import defaultdict
import gc
import functools
from tqdm import tqdm

from bitstack.utils.data_utils import get_loaders
from bitstack.modules.BitStackLinear import ScaledActivation

def get_op_name(module, op):
    # get the name of the op relative to the module
    for name, m in module.named_modules():
        if m is op:
            return name
    raise ValueError(f"Cannot find op {op} in module {module}")


def append_str_prefix(x, prefix):
    if isinstance(x, str):
        return prefix + x
    elif isinstance(x, tuple):
        return tuple([append_str_prefix(y, prefix) for y in x])
    elif isinstance(x, list):
        return [append_str_prefix(y, prefix) for y in x]
    else:
        return x

def get_op_by_name(module, op_name):
    # get the op by its name relative to the module
    for name, m in module.named_modules():
        if name == op_name:
            return m
    raise ValueError(f"Cannot find op {op_name} in module {module}")


def set_op_by_name(layer, name, new_module):
    levels = name.split(".")
    if len(levels) > 1:
        mod_ = layer
        for l_idx in range(len(levels) - 1):
            if levels[l_idx].isdigit():
                mod_ = mod_[int(levels[l_idx])]
            else:
                mod_ = getattr(mod_, levels[l_idx])
        setattr(mod_, levels[-1], new_module)
    else:
        setattr(layer, name, new_module)


@torch.no_grad()
def scale_ln_fcs(ln, fcs, scales, fuse_scales=True):
    if not isinstance(fcs, list):
        fcs = [fcs]

    scales = scales.to(ln.weight.device)
    
    if fuse_scales:
        ln.weight.div_(scales)
        if hasattr(ln, "bias") and ln.bias is not None:
            ln.bias.div_(scales)

    for fc in fcs:
        fc.weight.mul_(scales.view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1, fc2, scales, fuse_scales=True):
    # assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)
    # assert fc1.out_features == fc2.in_features

    scales = scales.to(fc1.weight.device)
    if fuse_scales:
        # fc1.weight.div_(scales.view(-1, 1))
        fc1.weight[-scales.size(0) :].div_(scales.view(-1, 1))
        if fc1.bias is not None:
            fc1.bias.div_(scales.view(-1))

    fc2.weight.mul_(scales.view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0

@torch.no_grad()
def get_act_scale(x):
    eps = 1e-3
    return torch.norm(x, dim=1).mean(dim=0) + eps


@torch.no_grad()
def auto_scale_block(module, module_kwargs, input_feat, w_bit, signed, k):

    if "use_cache" in module_kwargs:
        module_kwargs.pop("use_cache")


    def get_scale(prev_op, layers, inp, module2inspect=None):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        inp = inp.to(next(module2inspect.parameters()).device)
        scales = get_act_scale(inp).detach().cpu()

        return (
            get_op_name(module, prev_op),
            tuple([get_op_name(module, m) for m in layers]),
            scales,
        )
    
    scales_list = []  # return the searched scales

    if isinstance(module, (LlamaDecoderLayer, MistralDecoderLayer, Qwen2DecoderLayer)):
        # attention input
        scales_list.append(
            get_scale(
                prev_op=module.input_layernorm,
                layers=[
                    module.self_attn.q_proj,
                    module.self_attn.k_proj,
                    module.self_attn.v_proj,
                ],
                inp=input_feat["self_attn.q_proj"],
                module2inspect=module.self_attn
            )
        )
        # attn out
        # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
        if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
            scales_list.append(
                get_scale(
                    prev_op=module.self_attn.v_proj,
                    layers=[module.self_attn.o_proj],
                    inp=input_feat["self_attn.o_proj"],
                )
            )
        # fc1
        scales_list.append(
            get_scale(
                prev_op=module.post_attention_layernorm,
                layers=[module.mlp.gate_proj, module.mlp.up_proj],
                inp=input_feat["mlp.gate_proj"],
                module2inspect=module.mlp,
            )
        )
        # fc2
        scales_list.append(
            get_scale(
                prev_op=module.mlp.up_proj,
                layers=[module.mlp.down_proj],
                inp=input_feat["mlp.down_proj"],
            )
        )
    else:
        raise NotImplementedError(f"{type(module)} not supported yet!")
    
    return scales_list

def apply_scale(module, scales_list, input_feat_dict=None, fuse_scales=True):
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        prev_op.cuda()
        for layer in layers:
            layer.cuda()
        scales.cuda()

        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            # print('scale_fc_fc', prev_op_name, layer_names)
            if not fuse_scales:
                new_module = ScaledActivation(prev_op, scales)
                set_op_by_name(module, prev_op_name, new_module)
            scale_fc_fc(prev_op, layers[0], scales, fuse_scales)
        elif isinstance(prev_op, (nn.LayerNorm, LlamaRMSNorm, MistralRMSNorm, Qwen2RMSNorm)):
            # print('scale_ln_fcs', prev_op_name, layer_names)
            if not fuse_scales:
                new_module = ScaledActivation(prev_op, scales)
                set_op_by_name(module, prev_op_name, new_module)
            scale_ln_fcs(prev_op, layers, scales, fuse_scales)
        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # apply the scaling to input feat if given; prepare it for clipping
        if input_feat_dict is not None:
            for layer_name in layer_names:
                inp = input_feat_dict[layer_name]
                inp.div_(scales.view(1, -1).to(inp.device))

        prev_op.cpu()
        for layer in layers:
            layer.cpu()
        scales.cpu()


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}

def get_blocks(model):
    if model.__class__.__name__ == "LlamaForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "MistralForCausalLM":
        layers = model.model.layers
    elif model.__class__.__name__ == "Qwen2ForCausalLM":
        layers = model.model.layers
    else:
        raise NotImplementedError(type(model))
    return layers

def move_embed(model, device):
    if isinstance(model, (LlamaForCausalLM, MistralForCausalLM, Qwen2ForCausalLM)):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
    else:
        raise NotImplementedError(type(model))

def move_position_embed(model, device):
    if isinstance(model, (LlamaForCausalLM, Qwen2ForCausalLM)):
        model.model.rotary_emb = model.model.rotary_emb.to(device)
    elif isinstance(model, MistralForCausalLM):
        pass
    else:
        raise NotImplementedError(type(model))

@torch.no_grad()
def scale_model(model, tokenizer, calib_dataset, seed, nsamples, seqlen, batch_size, w_bit, k, signed, fuse_scales):
    samples = get_loaders(name=calib_dataset, nsamples=nsamples, seed=seed, seqlen=seqlen, tokenizer=tokenizer)
    samples = torch.cat(samples, dim=0)
    layers = get_blocks(model)
    layers[0] = layers[0].cuda()
    move_embed(model, "cuda")
    move_position_embed(model, "cuda")
    inps = []
    layer_kwargs = {}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp)
            layer_kwargs.update(kwargs)
            raise ValueError  # early exit to break later inference
    layers[0] = Catcher(layers[0])
    try:
        model(samples.to(next(model.parameters()).device))
    except ValueError:  # work with early exit
        pass
    del samples
    layers[0] = layers[0].module  # restore
    inps = inps[0]
    layers[0] = layers[0].cpu()
    move_embed(model, "cpu")
    move_position_embed(model, "cpu")
    gc.collect()
    torch.cuda.empty_cache()

    scales = []
    for i in tqdm(range(len(layers)), desc="Scaling"):
        layer = layers[i]
        layer = layer.cuda()
        named_linears = get_named_linears(layer)

        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)
        input_feat = defaultdict(list)
        handles = []
        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        device = next(layer.parameters()).device
        outputs = []
        for j in range(0, inps.size(0), batch_size):
            batch_inps = inps[j:j + batch_size].to(device)
            batch_out = layer(batch_inps, **layer_kwargs)[0]
            outputs.append(batch_out)
        inps = torch.cat(outputs, dim=0)

        for h in handles:
            h.remove()
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}
        torch.cuda.empty_cache()
        # print(input_feat)
        scales_list = auto_scale_block(
            layer,
            layer_kwargs,
            input_feat=input_feat,
            w_bit=w_bit,
            k=k,
            signed=signed
        )
        apply_scale(layers[i], scales_list, input_feat_dict=input_feat, fuse_scales=fuse_scales)
        # append prefix to make names global
        scales += append_str_prefix(
            scales_list, get_op_name(model, layer) + "."
        )
        torch.cuda.empty_cache()

        layer = layer.cpu()
        del input_feat
        gc.collect()
        torch.cuda.empty_cache()

    return scales