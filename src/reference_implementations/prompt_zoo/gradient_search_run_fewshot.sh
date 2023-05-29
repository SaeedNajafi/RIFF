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


EXPERIMENT_TYPE=${EXP_TYPE}
RANDOM_SEED=${SEED}
TASK_NAME=${TASK}
MAIN_PATH=${MAIN_PATH}
DATA_AUG=${AUG}
LOAD_PARAPHRASER=${LOAD_PARA}

model_path=${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}/${RANDOM_SEED}/${DATA_AUG}_${LOAD_PARAPHRASER}
mkdir -p ${MAIN_PATH}
mkdir -p ${MAIN_PATH}/${TASK_NAME}
mkdir -p ${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}
mkdir -p ${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}/${RANDOM_SEED}
mkdir -p ${model_path}

# train phase

CUDA_VISIBLE_DEVICES=${SLURM_PROCID} python -m src.reference_implementations.prompt_zoo.trainer \
    --train_batch_size 2 \
    --eval_batch_size 32 \
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
    --para_model_path /scratch/ssd004/scratch/snajafi/data_temp/roberta-exps/sst2/pg_z_score_para_model/${RANDOM_SEED} \
    --checkpoint best_step \
    --max_epochs 10 \
    --training_steps 1000000 \
    --steps_per_checkpoint 2 \
    --source_max_length 128 \
    --decoder_max_length 128 \
    --instruction_type manual_template_research_sst2_no_instruction \
    --pretrained_model roberta-large \
    --g_beam_size 1 \
    --top_k 20 \
    --enable_data_augmentation ${DATA_AUG} \
    --load_paraphraser ${LOAD_PARAPHRASER} \
    --beam_size 8 \
    --ensemble_type no_ensemble


# test phase
CUDA_VISIBLE_DEVICES=${SLURM_PROCID} python -m src.reference_implementations.prompt_zoo.trainer \
        --eval_batch_size 32 \
        --mode test \
        --seed ${RANDOM_SEED} \
        --task_name sst2 \
        --test_file validation \
        --num_classes 2 \
        --fewshot_sample_size 128 \
        --exp_type ${EXP_TYPE} \
        --model_path ${model_path} \
        --para_model_path /scratch/ssd004/scratch/snajafi/data_temp/roberta-exps/sst2/pg_z_score_para_model/${RANDOM_SEED} \
        --source_max_length 128 \
        --decoder_max_length 128 \
        --checkpoint best_step \
        --prediction_file ${model_path}/sst2.validation.gradient_search.no_ensemble.${EXP_TYPE}.predictions.csv \
        --instruction_type manual_template_research_sst2_no_instruction \
        --pretrained_model roberta-large \
        --enable_data_augmentation ${DATA_AUG} \
        --load_paraphraser ${LOAD_PARAPHRASER} \
        --beam_size 8 \
        --ensemble_type no_ensemble
