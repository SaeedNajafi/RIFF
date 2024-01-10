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
    checkpoint_path=/scratch/ssd004/scratch/snajafi/checkpoint
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
    cp -r ${PARA_MODEL_PATH}/para_t5_model_best_step ${model_path}/
fi

data_path=/h/snajafi/codes/paraphrase_inputs_for_prompts/16-shot
train_file=${data_path}/${TASK_NAME}/16-${RANDOM_SEED}/train.tsv
dev_file=${data_path}/${TASK_NAME}/16-${RANDOM_SEED}/dev.tsv
test_file=${data_path}/${TASK_NAME}/16-${RANDOM_SEED}/test.tsv

instruction_type=manual_template_research_${TASK_NAME}_with_instruction
train_batch_size=8
eval_batch_size=8
max_seq_len=128
test_sample_size=8
max_epochs=100
if [ "${EXPERIMENT_TYPE}" = "gradient_search" ]; then
    instruction_type=manual_template_research_${TASK_NAME}_no_instruction
    train_batch_size=2
    eval_batch_size=2
fi

ensembling="paraphrase_predict"
METRIC_TO_SAVE="all_accuracy"
if [ "${DATA_AUG}" = "0" ]; then
    ensembling="no_ensemble"
    METRIC_TO_SAVE="original_accuracy"
fi


# train phase
python -m src.reference_implementations.prompt_zoo.trainer \
    --train_batch_size ${train_batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --mode train \
    --seed ${RANDOM_SEED} \
    --task_name ${TASK_NAME} \
    --train_file ${train_file} \
    --dev_file ${dev_file} \
    --classification_type fewshot \
    --num_classes ${NUM_CLASSES} \
    --fewshot_sample_size ${FEWSHOT_SIZE} \
    --exp_type ${EXPERIMENT_TYPE} \
    --model_path ${model_path} \
    --para_model_path ${model_path} \
    --checkpoint best_step \
    --max_epochs ${max_epochs} \
    --learning_rate ${LEARN_RATE} \
    --training_steps 1000000 \
    --steps_per_checkpoint 8 \
    --source_max_length ${max_seq_len} \
    --decoder_max_length ${max_seq_len} \
    --weight_decay_rate 0.0001 \
    --instruction_type ${instruction_type} \
    --enable_data_augmentation ${DATA_AUG} \
    --enable_paraphrase_training ${TRAIN_PARAPHRASER} \
    --load_paraphraser ${LOAD_PARAPHRASER} \
    --ensemble_type ${ensembling} \
    --test_temperature 1.0 \
    --test_sample_size ${test_sample_size} \
    --metric_to_save ${METRIC_TO_SAVE} \
    --g_beam_size 1 \
    --top_k 4 \
    --test_sampling_algorithm "beam_search" \
    --use_cache 1 \
    --lm_type "llama2"

# test phase
python -m src.reference_implementations.prompt_zoo.trainer \
    --eval_batch_size ${eval_batch_size} \
    --mode test \
    --seed ${RANDOM_SEED} \
    --task_name ${TASK_NAME} \
    --test_file ${test_file} \
    --num_classes ${NUM_CLASSES} \
    --fewshot_sample_size ${FEWSHOT_SIZE} \
    --exp_type ${EXPERIMENT_TYPE} \
    --model_path ${model_path} \
    --para_model_path ${model_path} \
    --source_max_length ${max_seq_len} \
    --decoder_max_length ${max_seq_len} \
    --checkpoint best_step \
    --prediction_file ${model_path}/${TASK_NAME}.validation.with_instruction.${EXPERIMENT_TYPE}.all_predictions.csv \
    --instruction_type ${instruction_type} \
    --enable_data_augmentation ${DATA_AUG} \
    --enable_paraphrase_training ${TRAIN_PARAPHRASER} \
    --load_paraphraser ${LOAD_PARAPHRASER} \
    --ensemble_type ${ensembling} \
    --test_temperature 1.0 \
    --test_sample_size ${test_sample_size} \
    --g_beam_size 1 \
    --top_k 4 \
    --test_sampling_algorithm "beam_search" \
    --use_cache 1 \
    --lm_type "llama2"

rm -r -f ${model_path}/llama2_model_best_step
