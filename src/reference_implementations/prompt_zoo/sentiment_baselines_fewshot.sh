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
DATA_AUG=${AUG}
TRAIN_PARAPHRASER=${TRAIN_PARA}
LOAD_PARAPHRASER=${LOAD_PARA}

model_path=${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}/${RANDOM_SEED}/${LEARN_RATE}/${DATA_AUG}_${TRAIN_PARAPHRASER}_${LOAD_PARAPHRASER}
mkdir -p ${MAIN_PATH}
mkdir -p ${MAIN_PATH}/${TASK_NAME}
mkdir -p ${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}
mkdir -p ${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}/${RANDOM_SEED}
mkdir -p ${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}/${RANDOM_SEED}/${LEARN_RATE}
mkdir -p ${model_path}

if [ "${EXP_TYPE}" = "classifier_finetune" ]; then
    instruction_type="no_instruction"

elif [ "${TASK_NAME}" = "sst2" ]; then
    instruction_type="manual_template_research_sst2_with_instruction"

elif [ "${TASK_NAME}" = "SetFit/sst5" ]; then
    instruction_type="manual_template_research_sst5_with_instruction"

fi

# train phase
python -m src.reference_implementations.prompt_zoo.trainer \
    --train_batch_size 4 \
    --eval_batch_size 32 \
    --mode train \
    --seed ${RANDOM_SEED} \
    --task_name sst2 \
    --train_file train \
    --dev_file validation \
    --classification_type fewshot \
    --num_classes 2 \
    --exp_type ${EXP_TYPE} \
    --model_path ${model_path} \
    --para_model_path /scratch/ssd004/scratch/snajafi/data_temp/roberta-exps/sst2/pg_z_score_para_model/${RANDOM_SEED} \
    --checkpoint best_step \
    --max_epochs 10 \
    --learning_rate ${LEARN_RATE} \
    --training_steps 1000000 \
    --steps_per_checkpoint 4 \
    --source_max_length 128 \
    --decoder_max_length 128 \
    --weight_decay_rate 0.01 \
    --instruction_type ${instruction_type} \
    --pretrained_model roberta-large \
    --enable_data_augmentation ${DATA_AUG} \
    --enable_paraphrase_training ${TRAIN_PARAPHRASER} \
    --load_paraphraser ${LOAD_PARAPHRASER} \
    --beam_size 8 \
    --ensemble_type no_ensemble

# test phase
python -m src.reference_implementations.prompt_zoo.trainer \
    --eval_batch_size 32 \
    --mode test \
    --seed ${RANDOM_SEED} \
    --task_name sst2 \
    --test_file validation \
    --num_classes 2 \
    --exp_type ${EXP_TYPE} \
    --model_path ${model_path} \
    --para_model_path /scratch/ssd004/scratch/snajafi/data_temp/roberta-exps/sst2/pg_z_score_para_model/${RANDOM_SEED} \
    --source_max_length 128 \
    --decoder_max_length 128 \
    --checkpoint best_step \
    --prediction_file ${model_path}/sst2.validation.with_instruction.${EXP_TYPE}.no_ensemble.predictions.csv \
    --instruction_type ${instruction_type} \
    --pretrained_model roberta-large \
    --enable_data_augmentation ${DATA_AUG} \
    --enable_paraphrase_training ${TRAIN_PARAPHRASER} \
    --load_paraphraser ${LOAD_PARAPHRASER} \
    --beam_size 8 \
    --ensemble_type no_ensemble
