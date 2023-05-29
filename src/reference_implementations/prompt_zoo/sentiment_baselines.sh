#!/bin/bash

# For reading key=value arguments
for ARGUMENT in "$@"
do
	KEY=$(echo $ARGUMENT | cut -f1 -d=)
	KEY_LENGTH=${#KEY}
	VALUE="${ARGUMENT:$KEY_LENGTH+1}"
	export "$KEY"="$VALUE"
done

PROJECT_DIR=$( dirname -- "$0"; )

# We source to keep the internal env variables defined.
source ${PROJECT_DIR}/../setup_gpu_worker.sh

LEARN_RATE=${LR}
EXPERIMENT_TYPE=${EXP_TYPE}
RANDOM_SEED=${SEED}
TASK_NAME=${TASK}
MAIN_PATH=${MAIN_PATH}

model_path=${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}/${RANDOM_SEED}/${LEARN_RATE}
mkdir -p ${MAIN_PATH}
mkdir -p ${MAIN_PATH}/${TASK_NAME}
mkdir -p ${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}
mkdir -p ${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}/${RANDOM_SEED}
mkdir -p ${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}/${RANDOM_SEED}/${LEARN_RATE}
mkdir -p ${model_path}

# train phase
if [ "${TASK_NAME}" = "sst2" ]; then
    python -m src.reference_implementations.prompt_zoo.trainer \
        --train_batch_size 32 \
        --eval_batch_size 64 \
        --mode train \
        --seed ${RANDOM_SEED} \
        --task_name sst2 \
        --train_file train \
        --dev_file validation \
        --classification_type fullshot \
        --num_classes 2 \
        --exp_type ${EXP_TYPE} \
        --model_path ${model_path} \
        --max_epochs 5 \
        --learning_rate ${LEARN_RATE} \
        --training_steps 1000000 \
        --steps_per_checkpoint 10 \
        --source_max_length 128 \
        --decoder_max_length 128 \
        --weight_decay_rate 0.001 \
        --instruction_type manual_template_research_sst2_with_instruction \
        --pretrained_model roberta-large

elif [ "${TASK_NAME}" = "SetFit/sst5" ]; then
    python -m src.reference_implementations.prompt_zoo.trainer \
        --train_batch_size 32 \
        --eval_batch_size 64 \
        --mode train \
        --seed ${RANDOM_SEED} \
        --task_name SetFit/sst5 \
        --train_file train \
        --dev_file validation \
        --classification_type fullshot \
        --num_classes 5 \
        --exp_type ${EXP_TYPE} \
        --model_path ${model_path} \
        --max_epochs 5 \
        --learning_rate ${LEARN_RATE} \
        --training_steps 1000000 \
        --steps_per_checkpoint 10 \
        --source_max_length 128 \
        --decoder_max_length 128 \
        --weight_decay_rate 0.001 \
        --instruction_type manual_template_research_sst5_with_instruction \
        --pretrained_model roberta-large
fi

# test phase
if [ "${TASK_NAME}" = "sst2" ]; then
    python -m src.reference_implementations.prompt_zoo.trainer \
        --eval_batch_size 64 \
        --mode test \
        --seed ${RANDOM_SEED} \
        --task_name sst2 \
        --test_file validation \
        --num_classes 2 \
        --exp_type ${EXP_TYPE} \
        --model_path ${model_path} \
        --source_max_length 128 \
        --decoder_max_length 128 \
        --checkpoint best_step \
        --prediction_file ${model_path}/sst2.validation.with_instruction.${EXP_TYPE}.predictions.csv \
        --instruction_type manual_template_research_sst2_with_instruction \
        --pretrained_model roberta-large

elif [ "${TASK_NAME}" = "SetFit/sst5" ]; then
    python -m src.reference_implementations.prompt_zoo.trainer \
        --eval_batch_size 64 \
        --mode test \
        --seed ${RANDOM_SEED} \
        --task_name SetFit/sst5 \
        --test_file validation \
        --num_classes 5 \
        --exp_type ${EXP_TYPE} \
        --model_path ${model_path} \
        --source_max_length 128 \
        --decoder_max_length 128 \
        --checkpoint best_step \
        --prediction_file ${model_path}/sst5.validation.with_instruction.${EXP_TYPE}.predictions.csv \
        --instruction_type manual_template_research_sst5_with_instruction \
        --pretrained_model roberta-large

fi
