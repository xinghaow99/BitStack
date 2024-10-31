#!/bin/bash

python -m bitstack.main \
    --model_name_or_path outputs/Meta-Llama-3.1-8B_niter_16_k_16_no_avd_False_scaled_True \
    --k 16 \
    --max_memory_MB 5541 \
    --load_compressed \
    --do_eval \
    --lm_eval \
    --output_dir outputs

