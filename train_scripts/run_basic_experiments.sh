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

# run for all these tasks.
num_classes=(6)
tasks=(trec)

# rates=(0.0001 0.00001 0.001 0.001 0.001 0.001 0.5)
# exps=(lora_finetune all_finetune input_finetune output_finetune soft_prompt_finetune classifier_finetune gradient_search)

rates=(0.0001)
exps=(lora_finetune)
augs=(0)

for t in ${!tasks[@]};
do
    task=${tasks[$t]}
    num_class=${num_classes[$t]}
    for i in ${!rates[@]};
    do
        rate=${rates[$i]}
        exp=${exps[$i]}
        for g in ${!augs[@]};
        do
            aug=${augs[$g]}
            TOKENIZERS_PARALLELISM=false bash train_scripts/run_augmentation_experiments.sh \
                AUG=${aug} LR=${rate} \
                EXP_TYPE=${exp} TASK=${task} \
                FEWSHOT_SIZE=16 CLUSTER_NAME=vcluster NUM_CLASSES=${num_class} \
                GPU_TYPE=${GPU_TYPE}
        done
    done
done
