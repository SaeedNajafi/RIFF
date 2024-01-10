#!/bin/bash

# For reading key=value arguments
for ARGUMENT in "$@"
do
	KEY=$(echo $ARGUMENT | cut -f1 -d=)
	KEY_LENGTH=${#KEY}
	VALUE="${ARGUMENT:$KEY_LENGTH+1}"
	export "$KEY"="$VALUE"
done

TIME=${TIME}
LR=${LR}
EXP_TYPE=${EXP_TYPE}
TASK=${TASK}
AUG=${AUG}
FEWSHOT_SIZE=${FEWSHOT_SIZE}
CLUSTER_NAME=${CLUSTER_NAME}
NUM_CLASSES=${NUM_CLASSES}
GPU_TYPE=${GPU_TYPE}

# seeds=(12321 42 2023 11 1993)
# seeds used internally.

seeds=(100 13 21 42 87)

for s in ${!seeds[@]};
do
    seed=${seeds[$s]}
    bash src/reference_implementations/run_prompt.sh \
        TIME=${TIME} \
        SCRIPT=src/reference_implementations/prompt_zoo/test_fewshot_data_augmentation.sh \
        LOG_DIR=./roberta-exps-logs \
        EXP_TYPE=${EXP_TYPE} \
        TASK=${TASK} \
        SEED=${seed} \
        NUM_CLASSES=${NUM_CLASSES} \
        FEWSHOT_SIZE=${FEWSHOT_SIZE} \
        LR=${LR} \
        AUG=1 \
        TRAIN_PARA=0 \
        LOAD_PARA=1 \
        LEN=25 \
        PARA_LOSS="dummy" \
        SAMPLING_METHOD="dummy" \
        SAMPLING_ALG="dummy" \
        METRIC_TO_SAVE=original_accuracy \
        KL_COEFFICIENT=0.1 \
        CLUSTER_NAME=${CLUSTER_NAME} \
        GPU_TYPE=${GPU_TYPE} \
        PARA_MODEL_PATH=/checkpoint/snajafi/${TASK}-better-results/${TASK}_${NUM_CLASSES}_16_lora_finetune_${seed}_0.0001_25_1_0_1_original_accuracy \

done
