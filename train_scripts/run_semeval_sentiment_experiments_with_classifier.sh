#!/bin/bash

rates=(0.25 0.5 0.1 0.01 0.001)

# we use a dummy prompt length 100

for i in ${!rates[@]};
do
	rate=${rates[$i]}
    sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
            src/reference_implementations/prompt_zoo/train_semeval_sentiment.sh \
            ./torch-prompt-tuning-exps-logs \
            classifier_finetune \
            ${rate} \
            100
done
