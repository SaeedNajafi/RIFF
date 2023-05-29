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

EXPERIMENT_TYPE=${EXP_TYPE}
RANDOM_SEED=${SEED}
TASK_NAME=${TASK}
MAIN_PATH=${MAIN_PATH}

model_path=${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}/${RANDOM_SEED}
mkdir -p ${MAIN_PATH}
mkdir -p ${MAIN_PATH}/${TASK_NAME}
mkdir -p ${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}
mkdir -p ${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}/${RANDOM_SEED}
mkdir -p ${model_path}

# train phase
if [ "${TASK_NAME}" = "sst2" ]; then
    python -m src.reference_implementations.prompt_zoo.trainer \
        --train_batch_size 8 \
        --eval_batch_size 128 \
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
        --training_steps 1000000 \
        --steps_per_checkpoint 2 \
        --source_max_length 128 \
        --decoder_max_length 128 \
        --instruction_type manual_template_research_sst2_no_instruction \
        --pretrained_model roberta-large \
        --beam_size 1 \
        --top_k 20

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
        --training_steps 1000000 \
        --steps_per_checkpoint 10 \
        --source_max_length 128 \
        --decoder_max_length 128 \
        --instruction_type manual_template_research_sst5_no_instruction \
        --pretrained_model roberta-large \
        --beam_size 1 \
        --top_k 100
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
        --prediction_file ${model_path}/sst2.validation.gradient_search.${EXP_TYPE}.predictions.csv \
        --instruction_type manual_template_research_sst2_no_instruction \
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
        --prediction_file ${model_path}/sst5.validation.gradient_search.${EXP_TYPE}.predictions.csv \
        --instruction_type manual_template_research_sst5_no_instruction \
        --pretrained_model roberta-large

fi
