#!/bin/bash

sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
        src/reference_implementations/prompt_zoo/no_finetune.sh \
        ./roberta-exps-logs \
