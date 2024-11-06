import argparse
import torch
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from accelerate import (
    init_empty_weights,
    load_checkpoint_and_dispatch,
    infer_auto_device_map,
    dispatch_model,
)

import logging
import os
import json

from bitstack.modules.BitStackLinear import BitStackLinear
from bitstack.utils.scale_utils import scale_model
from bitstack.utils.model_utils import (
    set_model_bits,
    calculate_memory_per_bit,
    visualize_compression_config,
    eval_ppl,
    sort_layers_and_bits_average,
    generate_configs,
    check_model_memory_excluding_linear,
    int_or_str,
    load_model_and_tokenizer,
    check_module_memory,
    retrieve_compression_config,
    load_bitstack_model_and_tokenizer,
    prepare_for_fused_forward,
    prepare_for_saving

)
from bitstack.utils.data_utils import get_loaders
from bitstack.utils.decompose import decompose

logging.getLogger("accelerate.utils.modeling").setLevel(logging.ERROR) # Suppress warnings for partial loading

def check_empty_weights(module, name=''):
    for child_name, child in module.named_children():
        check_empty_weights(child, name + '.' + child_name if name else child_name)
    
    for param_name, param in module.named_parameters(recurse=False):
        if param.is_meta:
            print(f"Empty weight found: {name + '.' + param_name if name else param_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--calib_dataset', type=str, default='wikitext2')
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--nsamples', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--k', type=int, default=1)
    parser.add_argument('--niter', type=int, default=1)
    parser.add_argument('--no_avd', action='store_true')
    parser.add_argument('--no_fuse_scale', action='store_false', dest='fuse_scale')
    parser.add_argument('--fused_level', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval_dataset', type=str, default='wikitext2')
    parser.add_argument('--no_scale', action='store_false', dest='scale_weight')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--load_bitstack', action='store_true')
    parser.add_argument('--max_memory_MB', type=int, default=None)
    parser.add_argument('--do_eval', action='store_true')
    parser.add_argument('--lm_eval', action='store_true')
    parser.add_argument('--lm_eval_batch_size', type=int_or_str, default='auto')
    parser.add_argument(
        '--tasks',
        nargs='+',
        default=["piqa", "hellaswag", "arc_easy", "arc_challenge", "winogrande", "lambada_openai"],
    )
    parser.add_argument('--do_save', action='store_true')
    parser.add_argument('--score_importance', action='store_true')
    parser.add_argument('--generate_compression_configs', action='store_true')
    parser.set_defaults(scale_weight=True, fuse_scale=True)
    args = parser.parse_args()

    if args.load_bitstack:
        compression_config = None
        if args.max_memory_MB is not None:
            with open(os.path.join(args.model_name_or_path, 'compression_config.json'), 'r') as f:
                compression_configs = json.load(f)
            compression_config = retrieve_compression_config(compression_configs, args.max_memory_MB)
        model, tokenizer, model_name = load_bitstack_model_and_tokenizer(args.model_name_or_path, args.niter, args.k, args.no_avd, args.fused_level, compression_config)
    else:
        model, tokenizer = load_model_and_tokenizer(args.model_name_or_path)
        model_name = args.model_name_or_path.split("/")[-1]
        if args.scale_weight:
            print("Scaling weights")
            scales = scale_model(model, tokenizer, args.calib_dataset, args.seed, args.nsamples, args.seqlen, args.batch_size, args.niter, args.k, args.no_avd, args.fuse_scale)
        decompose(model, args.niter, args.k, args.no_avd, args.fused_level, init_only=False)

    
    save_path = os.path.join(args.output_dir, f'{args.model_name_or_path.split("/")[-1]}_niter_{args.niter}_k_{args.k}_no_avd_{args.no_avd}_scaled_{args.scale_weight}')
    if args.do_save:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    if not args.load_bitstack:
        device_map = infer_auto_device_map(model, no_split_module_classes=['LlamaDecoderLayer'])
        model = dispatch_model(model, device_map=device_map)

    if args.fused_level > 0:
        prepare_for_fused_forward(model)

    if args.do_eval:
        saved_metrics = {}
        if args.max_memory_MB is not None:
            saved_metrics['max_memory (MB)'] = args.max_memory_MB
            saved_metrics['compression_config'] = compression_config

        saved_metrics['k'] = args.k
        model.eval()
        testloader = get_loaders(args.eval_dataset, seed=args.seed, tokenizer=tokenizer, seqlen=args.seqlen, eval_mode=True)
        ppl = eval_ppl(args.batch_size, args.seqlen, model, testloader)
        print(f"Perplexity: {ppl}")
        saved_metrics['ppl'] = ppl
        if args.lm_eval:
            import lm_eval
            from lm_eval.models.huggingface import HFLM
            hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=args.lm_eval_batch_size)
            results = lm_eval.simple_evaluate(hflm, tasks=args.tasks, batch_size=args.lm_eval_batch_size)['results']
            print(results)
            saved_metrics['lm_eval'] = results
        eval_save_path = os.path.join(args.output_dir, 'evals', model_name, f'bitstack_k_{args.k}_max_memory_{args.max_memory_MB}')
        os.makedirs(eval_save_path, exist_ok=True)
        if args.load_bitstack and compression_config:
            visualize_compression_config(compression_config, save_path=os.path.join(eval_save_path, 'compression_config.png'))
        with open(os.path.join(eval_save_path, 'metrics.json'), 'w') as f:
            json.dump(saved_metrics, f, indent=4)

    if args.score_importance:
        validloader = get_loaders(args.calib_dataset, nsamples=32, seed=args.seed+1, seqlen=args.seqlen, tokenizer=tokenizer, eval_mode=False)
        validloader = torch.cat(validloader, dim=0)
        original_ppl = eval_ppl(args.batch_size, args.seqlen, model, validloader, disable_tqdm=True)
        all_linears = []
        for name, module in model.named_modules():
            if isinstance(module, BitStackLinear):
                all_linears.append(module)
        ppls = []
        print(f"Original Perplexity: {original_ppl}")
        for bit in range(1, args.niter):
            for linear in all_linears: linear.set_bit(bit)
            pbar = tqdm(all_linears)
            bit_ppls = []
            for linear in pbar:
                linear.set_bit(bit+1)
                ppl = eval_ppl(args.batch_size, args.seqlen, model, validloader, disable_tqdm=True)
                pbar.set_description(f"{bit} bit: {linear.name} ppl: {ppl}")
                bit_ppls.append({'layer': linear.name, 'ppl': ppl})
                linear.set_bit(bit)
            ppls.append({'bit': bit, 'ppls': bit_ppls})
            # Save after every bit finished
            data = {'original_ppl': original_ppl, 'reduced_ppl': ppls}
            if args.load_bitstack:
                reduced_ppl_path = os.path.join(args.model_name_or_path, 'reduced_ppl_average.json')
            else:
                reduced_ppl_path = os.path.join(args.output_dir, f'{args.model_name_or_path.split("/")[-1]}_niter_{args.niter}_k_{args.k}_no_avd_{args.no_avd}_scaled_{args.scale_weight}', 'reduced_ppl_average.json')
            with open(reduced_ppl_path, 'w') as f:
                json.dump(data, f, indent=4)

    if args.generate_compression_configs:
        if args.load_bitstack:
            reduced_ppl_path = os.path.join(args.model_name_or_path, 'reduced_ppl_average.json')
        else:
            reduced_ppl_path = os.path.join(args.output_dir, f'{args.model_name_or_path.split("/")[-1]}_niter_{args.niter}_k_{args.k}_no_avd_{args.no_avd}_scaled_{args.scale_weight}', 'reduced_ppl_average.json')
        with open(reduced_ppl_path, 'r') as f:
            reduced_ppls = json.load(f)
        sorted_layer_bits = sort_layers_and_bits_average(reduced_ppls)
        extra_memory = check_model_memory_excluding_linear(model)
        memory_per_bit = calculate_memory_per_bit(model)
        minimum_memory = sum(memory_per_bit.values())
        total_memory = minimum_memory + extra_memory
        configs = generate_configs(sorted_layer_bits, memory_per_bit, total_memory)
        if args.load_bitstack:
            save_path = args.model_name_or_path
        with open(os.path.join(save_path, 'compression_config.json'), 'w') as f:
            json.dump(configs, f, indent=4)
if __name__ == '__main__':
    main()