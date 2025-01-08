from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import re
import torch
from tqdm import tqdm

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from bitstack.modules.BitStackLinear import BitStackLinear
from bitstack.utils.decompose import decompose

def set_model_bits(model, compression_config):
    # Set bits for each layer as specified in compression_config
    # This function does not actually save memory usage, only for evaluation
    for name, module in model.named_modules():
        if isinstance(module, BitStackLinear):
            module.set_bit(compression_config[module.name])
    return model

def calculate_memory_per_bit(model):
    memory_per_bit = {}
    for name, module in model.named_modules():
        if isinstance(module, BitStackLinear):
            memory_per_bit[module.name] = module.memory_per_bit()
    return memory_per_bit

def sort_layers_and_bits_average(data):
    bits = data['reduced_ppl']
    max_bit = len(bits)
    # preprocess for nans
    # for each bit, we first sort by non-nan ppl, then sort by next bit ppl for nan layers
    for bit in reversed(range(max_bit)):
        bit_data = bits[bit]
        bit_ppls = bit_data['ppls']
        max_ppl = max(bit_ppls, key=lambda x: x['ppl'])['ppl']
        if str(max_ppl).lower() == 'nan':
            max_ppl = 0
        for idx, bit_ppl in enumerate(bit_ppls):
            if str(bit_ppl['ppl']).lower() == 'nan':
                if bit == max_bit - 1:
                    raise ValueError("Last bit is nan")
                next_bit_ppls = bits[bit+1]['ppls']
                assert next_bit_ppls[idx]['layer'] == bit_ppl['layer']
                bit_ppl['ppl'] = next_bit_ppls[idx]['ppl']+max_ppl
                
    sorted_layers = []
    for b in range(max_bit):
        bit_data = bits[b]
        bit_ppls = bit_data['ppls']
        sorted_layers_per_bit = sorted(bit_ppls, key=lambda x: x['ppl'])
        sorted_layers.extend([layer['layer'] for layer in sorted_layers_per_bit])
    return sorted_layers

def generate_configs(sorted_layer_bits, memory_per_bit, total_memory):
    configs = []
    # print(memory_per_bit)
    # Basic config: every layer is 1 bit
    layers = {layer: 1 for layer in memory_per_bit.keys()}
    basic_config = {'memory': total_memory, 'layers': layers.copy()}
    configs.append(basic_config)
    for layer in sorted_layer_bits:
        total_memory += memory_per_bit[layer]
        layers[layer] += 1
        # print(f"Increased {layer} to {layers[layer]} bits")
        configs.append({'memory': total_memory, 'layers': layers.copy()})
        # print(configs[-1])
    return configs

def visualize_compression_config(compression_config, save_path):
    layers = list(compression_config.keys())
    bits = list(compression_config.values())

    # Define colors for different matrix types
    color_map = {
        'q_proj': 'red',
        'k_proj': 'green',
        'v_proj': 'blue',
        'o_proj': 'yellow',
        'gate_proj': 'purple',
        'up_proj': 'orange',
        'down_proj': 'cyan'
    }

    # Extract layer numbers and group bits by layer and matrix type
    layer_numbers = []
    grouped_bits = {}
    for layer, bit in zip(layers, bits):
        match = re.search(r'layer\.(\d+)\..*?(\w+_proj)$', layer)
        if match:
            layer_num = int(match.group(1))
            matrix_type = match.group(2)
            if layer_num not in grouped_bits:
                grouped_bits[layer_num] = {}
            grouped_bits[layer_num][matrix_type] = bit
            if layer_num not in layer_numbers:
                layer_numbers.append(layer_num)

    plt.figure(figsize=(20, 10))

    bar_width = 0.1
    index = range(len(layer_numbers))

    for i, matrix_type in enumerate(color_map.keys()):
        values = [grouped_bits[layer].get(matrix_type, 0) for layer in layer_numbers]
        plt.bar([x + i * bar_width for x in index], values, bar_width,
                color=color_map[matrix_type], label=matrix_type)

    plt.xlabel('Layer Number')
    plt.ylabel('Bits')
    plt.title('Compression Configuration Visualization')
    plt.xticks([x + 3 * bar_width for x in index], layer_numbers)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

@torch.no_grad()
def eval_ppl(batch_size, seqlen, model, input_ids, disable_tqdm=False):
    model.eval()
    input_ids = [input_ids[i:i + batch_size] for i in range(0, len(input_ids), batch_size)]
    nbatches = len(input_ids)
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    nlls = []
    for i in tqdm(range(nbatches), disable=disable_tqdm):
        inputs = input_ids[i].to(model.device)
        lm_logits = model(inputs).logits
        shift_logits = lm_logits[:, :-1, :] # [batch_size, seq_len - 1, vocab_size]
        shift_labels = inputs[:, 1:] # [batch_size, seq_len - 1]
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
        neg_log_likelihood = loss.float().mean(dim=1)
        nlls.append(neg_log_likelihood)
    nll_tensor = torch.cat(nlls)
    ppl = torch.exp(nll_tensor.mean())
    return ppl.item()

def check_model_memory_excluding_linear(model, decomposed_modules):
    total_memory = 0
    for name, p in model.named_parameters():
        if any(exclude in name for exclude in decomposed_modules):
            if 'bias' in name: # bias is not None
                total_memory += p.numel() * p.element_size()
            continue
        total_memory += p.numel() * p.element_size()
    for name, b in model.named_buffers():
        if any(exclude in name for exclude in decomposed_modules):
            if 'bias' in name: # bias is not None
                total_memory += b.numel() * b.element_size()
            continue
        total_memory += b.numel() * b.element_size()
    return total_memory

def int_or_str(value):
    try:
        return int(value)
    except ValueError:
        return value

def check_module_memory(module):
    total_memory = 0
    for p in module.parameters():
        total_memory += p.numel() * p.element_size()
    for p in module.buffers():
        total_memory += p.numel() * p.element_size()
    return total_memory

def convert_to_fp16(model):
    for name, param in model.named_parameters():
        if param.dtype == torch.bfloat16:
            param.data = param.data.to(torch.float16)
    return model

def load_model_and_tokenizer(model_name_or_path):
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, config=config, trust_remote_code=True, torch_dtype='auto', low_cpu_mem_usage=True)
    model.eval()
    return model, tokenizer

def retrieve_compression_config(configs, max_memory_MB):
    max_memory = max_memory_MB * 1024**2
    # Sort configs by memory in descending order
    sorted_configs = sorted(configs, key=lambda x: x['memory'], reverse=True)
    for config in sorted_configs:
        if config['memory'] <= max_memory:
            # print(f"Config with memory {config['memory']//1024**2} MB found")
            return config['layers']
    raise ValueError(f"No config found within the memory limit {max_memory_MB} MB")

def load_bitstack_model_and_tokenizer(model_name_or_path, niter=16, k=16, no_avd=False, no_split_module_classes=['LlamaDecoderLayer'], fused_level=0, compression_config=None):
    config = AutoConfig.from_pretrained(model_name_or_path)
    model_name = config._name_or_path.split("/")[-1].split("_")[0]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    print(f"Loading decomposed model from {model_name_or_path}")
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float16)
    assert not (niter is None and compression_config is None)
    decompose(model, niter, k, no_avd, fused_level, init_only=True, compression_config=compression_config)
    load_checkpoint_and_dispatch(model, model_name_or_path, device_map='auto', no_split_module_classes=no_split_module_classes)
    # print memory usage
    memory = check_module_memory(model)
    torch.cuda.empty_cache()
    print(f"Memory: {memory//1024**2} MB")
    if fused_level > 0:
        prepare_for_fused_forward(model)
    return model, tokenizer, model_name

def prepare_for_fused_forward(model):
    print("Preparing for fused forward")
    for name, module in model.named_modules():
        if isinstance(module, BitStackLinear):
            module.stack_blocks()

def prepare_for_saving(model):
    for name, module in model.named_modules():
        if isinstance(module, BitStackLinear):
            module.prepare_for_saving()