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

#seeds=(11 42 1993 2023 12321)
seeds=(1993)

for s in ${!seeds[@]};
do
    seed=${seeds[$s]}
    bash src/reference_implementations/run_prompt.sh \
        TIME=${TIME} \
        SCRIPT=src/reference_implementations/prompt_zoo/fewshot_data_augmentation.sh \
        LOG_DIR=./roberta-exps-logs \
        EXP_TYPE=${EXP_TYPE} \
        TASK=${TASK} \
        SEED=${seed} \
        NUM_CLASSES=${NUM_CLASSES} \
        FEWSHOT_SIZE=${FEWSHOT_SIZE} \
        LR=${LR} \
        AUG=${AUG} \
        TRAIN_PARA=0 \
        LOAD_PARA=0 \
        LEN=25 \
        PARA_LOSS="dummy" \
        SAMPLING_METHOD="dummy" \
        SAMPLING_ALG="dummy" \
        METRIC_TO_SAVE=original_accuracy \
        KL_COEFFICIENT=0.0 \
        CLUSTER_NAME=${CLUSTER_NAME}
done
