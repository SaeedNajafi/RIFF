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
NUM_CLASSES=${NUM_CLASSES}
FEWSHOT_SIZE=${FEWSHOT_SIZE}
DATA_AUG=${AUG}
TRAIN_PARAPHRASER=${TRAIN_PARA}
LOAD_PARAPHRASER=${LOAD_PARA}
SLURM_JOB_ID=${SLURM_JOB_ID}
METRIC_TO_SAVE=${METRIC_TO_SAVE}
PROMPT_LEN=${LEN}
PARA_MODEL_PATH=${PARA_MODEL_PATH}
CLUSTER_NAME=${CLUSTER_NAME}

# We source the python env inside a worker depending on the cluster.
source ${PROJECT_DIR}/../setup_gpu_worker.sh CLUSTER_NAME=${CLUSTER_NAME}

if [ "${CLUSTER_NAME}" = "narval" ]; then
    checkpoint_path=/home/$USER/scratch/checkpoint/${SLURM_JOB_ID}

elif [ "${CLUSTER_NAME}" = "vcluster" ]; then
    checkpoint_path=/checkpoint/$USER/${SLURM_JOB_ID}

elif [ "${CLUSTER_NAME}" = "linux" ]; then
    checkpoint_path=~/checkpoint
fi

# create checkpoint path if it doesn't exist.
mkdir -p ${checkpoint_path}

experiment_name=${TASK_NAME}_${NUM_CLASSES}_${FEWSHOT_SIZE}_${EXPERIMENT_TYPE}_${RANDOM_SEED}
experiment_name=${experiment_name}_${LEARN_RATE}_${PROMPT_LEN}_${DATA_AUG}_${TRAIN_PARAPHRASER}_${LOAD_PARAPHRASER}_${METRIC_TO_SAVE}

model_path=${checkpoint_path}/${experiment_name}

mkdir -p ${model_path}

# delay purge in the checkpoint and job_id, required for vcluster.
touch ${checkpoint_path}/DELAYPURGE

if [ "${LOAD_PARAPHRASER}" = "1" ]; then
    cp -r ${PARA_MODEL_PATH}/bart_model_best_step ${model_path}/
fi

if [ "${TASK_NAME}" = "sst2" ]; then

    instruction_type="manual_template_research_sst2_with_instruction"
    train_batch_size=4
    if [ "${EXPERIMENT_TYPE}" = "gradient_search" ]; then
        instruction_type="manual_template_research_sst2_no_instruction"
        train_batch_size=4
    fi

elif [ "${TASK_NAME}" = "SetFit_sst5" ]; then
    instruction_type="manual_template_research_sst5_with_instruction"
    train_batch_size=4
    if [ "${EXPERIMENT_TYPE}" = "gradient_search" ]; then
        instruction_type="manual_template_research_sst5_no_instruction"
        train_batch_size=4
    fi
fi

ensembling="paraphrase_predict"
if [ "${DATA_AUG}" = "0" ]; then
        ensembling="no_ensemble"
fi

# train phase
python -m src.reference_implementations.prompt_zoo.trainer \
    --train_batch_size ${train_batch_size} \
    --eval_batch_size 8 \
    --mode train \
    --seed ${RANDOM_SEED} \
    --task_name ${TASK_NAME} \
    --train_file train \
    --dev_file validation \
    --classification_type fewshot \
    --num_classes ${NUM_CLASSES} \
    --fewshot_sample_size ${FEWSHOT_SIZE} \
    --exp_type ${EXPERIMENT_TYPE} \
    --model_path ${model_path} \
    --para_model_path ${model_path} \
    --checkpoint best_step \
    --max_epochs 100 \
    --learning_rate ${LEARN_RATE} \
    --training_steps 1000000 \
    --steps_per_checkpoint 8 \
    --source_max_length 128 \
    --decoder_max_length 128 \
    --weight_decay_rate 0.0001 \
    --instruction_type ${instruction_type} \
    --pretrained_model roberta-large \
    --enable_data_augmentation ${DATA_AUG} \
    --enable_paraphrase_training ${TRAIN_PARAPHRASER} \
    --load_paraphraser ${LOAD_PARAPHRASER} \
    --ensemble_type ${ensembling} \
    --test_temperature 1.0 \
    --test_sample_size 8 \
    --metric_to_save ${METRIC_TO_SAVE} \
    --g_beam_size 1 \
    --top_k 8 \
    --num_candidates 8 \
    --num_compose 1 \
    --meta_dir . \
    --meta_name search.txt \
    --level word

# test phase
python -m src.reference_implementations.prompt_zoo.trainer \
    --eval_batch_size 4 \
    --mode test \
    --seed ${RANDOM_SEED} \
    --task_name ${TASK_NAME} \
    --test_file validation \
    --num_classes ${NUM_CLASSES} \
    --fewshot_sample_size ${FEWSHOT_SIZE} \
    --exp_type ${EXPERIMENT_TYPE} \
    --model_path ${model_path} \
    --para_model_path ${model_path} \
    --source_max_length 128 \
    --decoder_max_length 128 \
    --checkpoint best_step \
    --prediction_file ${model_path}/${TASK_NAME}.validation.with_instruction.${EXPERIMENT_TYPE}.all_predictions.csv \
    --instruction_type ${instruction_type} \
    --pretrained_model roberta-large \
    --enable_data_augmentation ${DATA_AUG} \
    --enable_paraphrase_training ${TRAIN_PARAPHRASER} \
    --load_paraphraser ${LOAD_PARAPHRASER} \
    --ensemble_type ${ensembling} \
    --test_temperature 1.0 \
    --test_sample_size 8 \
    --g_beam_size 1 \
    --top_k 8 \
    --num_candidates 8 \
    --num_compose 1 \
    --meta_dir . \
    --meta_name search.txt \
    --level word
