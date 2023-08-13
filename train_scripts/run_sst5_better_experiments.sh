#!/bin/bash

rates=(0.0001 0.00001 0.001 0.00001 0.001 0.001 0.5)
exps=(lora_finetune all_finetune input_finetune output_finetune soft_prompt_finetune classifier_finetune gradient_search)

for i in ${!rates[@]};
do
    rate=${rates[$i]}
    exp=${exps[$i]}
    TOKENIZERS_PARALLELISM=false bash train_scripts/run_better_augmentation_experiments.sh \
        LR=${rate} \
        EXP_TYPE=${exp} TASK=SetFit_sst5 \
        FEWSHOT_SIZE=16 CLUSTER_NAME=vcluster NUM_CLASSES=5 \
        PARA_MODEL_PATH=/checkpoint/snajafi/sst5_trained_para_models
done
