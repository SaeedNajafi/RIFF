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

#rates=(0.0001 0.00001 0.001 0.00001 0.001)
#exps=(lora_finetune all_finetune input_finetune output_finetune soft_prompt_finetune)

rates=(0.5)
exps=(gradient_search)

for i in ${!rates[@]};
do
    rate=${rates[$i]}
    exp=${exps[$i]}
    TOKENIZERS_PARALLELISM=false bash train_scripts/run_better_augmentation_experiments.sh \
        LR=${rate} \
        EXP_TYPE=${exp} TASK=yelp_review_full \
        FEWSHOT_SIZE=16 CLUSTER_NAME=vcluster NUM_CLASSES=5 \
        PARA_MODEL_PATH=/checkpoint/snajafi/yelp-pretrained-models \
        GPU_TYPE=${GPU_TYPE}
done
