#!/bin/bash

# For reading key=value arguments
for ARGUMENT in "$@"
do
	KEY=$(echo $ARGUMENT | cut -f1 -d=)
	KEY_LENGTH=${#KEY}
	VALUE="${ARGUMENT:$KEY_LENGTH+1}"
	export "$KEY"="$VALUE"
done

GPU_TYPE=${GPU_TYPE}

rates=(0.0001 0.00001 0.001 0.00001 0.001 0.001)
exps=(lora_finetune all_finetune input_finetune output_finetune soft_prompt_finetune classifier_finetune)
augs=(0)

for i in ${!rates[@]};
do
    rate=${rates[$i]}
    exp=${exps[$i]}
    for g in ${!augs[@]};
    do
        aug=${augs[$g]}
        TOKENIZERS_PARALLELISM=false bash train_scripts/run_augmentation_experiments.sh \
            AUG=${aug} LR=${rate} \
            EXP_TYPE=${exp} TASK=SetFit_sst5 \
            FEWSHOT_SIZE=128 CLUSTER_NAME=vcluster NUM_CLASSES=5 \
            GPU_TYPE=${GPU_TYPE}
    done
done
