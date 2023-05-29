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

LEARN_RATE=${LR}
EXPERIMENT_TYPE=${EXP_TYPE}
RANDOM_SEED=${SEED}
TASK_NAME=${TASK}
MAIN_PATH=${MAIN_PATH}
PROMPT_LEN=${LEN}
DATA_AUG=${AUG}
TRAIN_PARAPHRASER=${TRAIN_PARA}
LOAD_PARAPHRASER=${LOAD_PARA}

model_path=${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}/${RANDOM_SEED}/${LEARN_RATE}/${PROMPT_LEN}/${DATA_AUG}_${TRAIN_PARAPHRASER}_${LOAD_PARAPHRASER}
mkdir -p ${MAIN_PATH}
mkdir -p ${MAIN_PATH}/${TASK_NAME}
mkdir -p ${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}
mkdir -p ${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}/${RANDOM_SEED}
mkdir -p ${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}/${RANDOM_SEED}/${LEARN_RATE}
mkdir -p ${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}/${RANDOM_SEED}/${LEARN_RATE}/${PROMPT_LEN}
mkdir -p ${model_path}

# train phase
CUDA_VISIBLE_DEVICES=${SLURM_PROCID} python -m src.reference_implementations.prompt_zoo.trainer \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --mode train \
    --seed ${RANDOM_SEED} \
    --task_name sst2 \
    --train_file train \
    --dev_file validation \
    --classification_type fewshot \
    --num_classes 2 \
    --fewshot_sample_size 128 \
    --exp_type ${EXP_TYPE} \
    --model_path ${model_path} \
    --checkpoint best_step \
    --max_epochs 100 \
    --learning_rate ${LEARN_RATE} \
    --training_steps 1000000 \
    --steps_per_checkpoint 32 \
    --source_max_length 128 \
    --decoder_max_length 128 \
    --weight_decay_rate 0.01 \
    --instruction_type manual_template_research_sst2_with_instruction \
    --pretrained_model roberta-large \
    --prompt_length ${PROMPT_LEN} \
    --enable_data_augmentation ${DATA_AUG} \
    --enable_paraphrase_training ${TRAIN_PARAPHRASER} \
    --load_paraphraser ${LOAD_PARAPHRASER} \
    --ensemble_type no_ensemble \
    --test_temperature 1.0 \
    --test_sample_size 8

# test phase
CUDA_VISIBLE_DEVICES=${SLURM_PROCID} python -m src.reference_implementations.prompt_zoo.trainer \
    --eval_batch_size 4 \
    --mode test \
    --seed ${RANDOM_SEED} \
    --task_name sst2 \
    --test_file validation \
    --num_classes 2 \
    --fewshot_sample_size 128 \
    --exp_type ${EXP_TYPE} \
    --model_path ${model_path} \
    --source_max_length 128 \
    --decoder_max_length 128 \
    --checkpoint best_step \
    --prediction_file ${model_path}/sst2.validation.with_instruction.${EXP_TYPE}.no_ensemble.predictions.csv \
    --instruction_type manual_template_research_sst2_with_instruction \
    --pretrained_model roberta-large \
    --prompt_length ${PROMPT_LEN} \
    --enable_data_augmentation ${DATA_AUG} \
    --enable_paraphrase_training ${TRAIN_PARAPHRASER} \
    --load_paraphraser ${LOAD_PARAPHRASER} \
    --ensemble_type no_ensemble \
    --test_temperature 1.0 \
    --test_sample_size 8

rm -r -f ${model_path}/roberta_model_best_step
