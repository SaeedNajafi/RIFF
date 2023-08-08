#!/bin/bash

# For reading key=value arguments
for ARGUMENT in "$@"
do
	KEY=$(echo $ARGUMENT | cut -f1 -d=)
	KEY_LENGTH=${#KEY}
	VALUE="${ARGUMENT:$KEY_LENGTH+1}"
	export "$KEY"="$VALUE"
done

CLUSTER_NAME=${CLUSTER_NAME}
EXP_TYPE=${EXP_TYPE}
TASK=${TASK}
SEED=${SEED}
NUM_CLASSES=${NUM_CLASSES}
FEWSHOT_SIZE=${FEWSHOT_SIZE}
LR=${LR}
AUG=${AUG}
TRAIN_PARA=${TRAIN_PARA}
LOAD_PARA=${LOAD_PARA}
LEN=${LEN}
PARA_LOSS=${PARA_LOSS}
SAMPLING_METHOD=${SAMPLING_METHOD}
SAMPLING_ALG=${SAMPLING_ALG}
METRIC_TO_SAVE=${METRIC_TO_SAVE}
KL_COEFFICIENT=${KL_COEFFICIENT}
PARA_MODEL_PATH=${PARA_MODEL_PATH}
TIME=${TIME}
SCRIPT=${SCRIPT}
LOG_DIR=${LOG_DIR}


if [ "${CLUSTER_NAME}" = "narval" ]; then
    sbatch \
        --account=def-afyshe-ab \
        --gres=gpu:a100:1 \
        --time=${TIME} \
        src/reference_implementations/run_prompt.slrm \
            ${SCRIPT} \
            ${LOG_DIR} \
            ${EXP_TYPE} \
            ${TASK} \
            ${SEED} \
            ${NUM_CLASSES} \
            ${FEWSHOT_SIZE} \
            ${LR} \
            ${AUG} \
            ${TRAIN_PARA} \
            ${LOAD_PARA} \
            ${LEN} \
            ${PARA_LOSS} \
            ${SAMPLING_METHOD} \
            ${SAMPLING_ALG} \
            ${METRIC_TO_SAVE} \
            ${KL_COEFFICIENT} \
            ${CLUSTER_NAME} \
            ${PARA_MODEL_PATH}


elif [ "${CLUSTER_NAME}" = "vcluster" ]; then
    sbatch \
        --gres=gpu:1 \
        --partition=a40 \
        --qos=normal \
        src/reference_implementations/run_prompt.slrm \
            ${SCRIPT} \
            ${LOG_DIR} \
            ${EXP_TYPE} \
            ${TASK} \
            ${SEED} \
            ${NUM_CLASSES} \
            ${FEWSHOT_SIZE} \
            ${LR} \
            ${AUG} \
            ${TRAIN_PARA} \
            ${LOAD_PARA} \
            ${LEN} \
            ${PARA_LOSS} \
            ${SAMPLING_METHOD} \
            ${SAMPLING_ALG} \
            ${METRIC_TO_SAVE} \
            ${KL_COEFFICIENT} \
            ${CLUSTER_NAME} \
            ${PARA_MODEL_PATH}


elif [ "${CLUSTER_NAME}" = "linux" ]; then
    bash ${SCRIPT} \
        EXP_TYPE=${EXP_TYPE} \
        TASK=${TASK} \
        SEED=${SEED} \
        NUM_CLASSES=${NUM_CLASSES} \
        FEWSHOT_SIZE=${FEWSHOT_SIZE} \
        LR=${LR} \
        AUG=${AUG} \
        TRAIN_PARA=${TRAIN_PARA} \
        LOAD_PARA=${LOAD_PARA} \
        LEN=${LEN} \
        PARA_LOSS=${PARA_LOSS} \
        SAMPLING_METHOD=${SAMPLING_METHOD} \
        SAMPLING_ALG=${SAMPLING_ALG} \
        METRIC_TO_SAVE=${METRIC_TO_SAVE} \
        KL_COEFFICIENT=${KL_COEFFICIENT} \
        PARA_MODEL_PATH=${PARA_MODEL_PATH} \
        CLUSTER_NAME=${CLUSTER_NAME}

fi
