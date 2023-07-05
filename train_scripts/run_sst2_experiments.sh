#!/bin/bash

rates=(0.0001 0.00001 0.001 0.00001 0.001 0.001 0.5)
exps=(lora_finetune all_finetune input_finetune output_finetune soft_prompt_finetune classifier_finetune gradient_search)

for i in ${!rates[@]};
do
    rate=${rates[$i]}
    exp=${exps[$i]}
    for g in ${!augs[@]};
    do
        aug=${augs[$g]}
        TOKENIZERS_PARALLELISM=false bash train_scripts/run_augmentation_experiments.sh \
            AUG=${aug} LR=${rate} \
            EXP_TYPE=${exp} TASK=sst2 \
            FEWSHOT_SIZE=32 CLUSTER_NAME=vcluster NUM_CLASSES=2
    done
done
