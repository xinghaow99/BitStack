#!/bin/bash

python -m bitstack.main \
    --model_name_or_path meta-llama/Meta-Llama-3.1-8B \
    --niter 16 \
    --k 16 \
    --nsamples 256 \
    --do_save \
    --output_dir outputs \
    --score_importance \
    --generate_compression_configs