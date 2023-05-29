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
PARA_LOSS=${PARA_LOSS}
SAMPLING_METHOD=${SAMPLING_METHOD}

model_path=${MAIN_PATH}/${TASK_NAME}/${EXPERIMENT_TYPE}/${RANDOM_SEED}/${LEARN_RATE}/${DATA_AUG}_${TRAIN_PARAPHRASER}_${LOAD_PARAPHRASER}_${PARA_LOSS}_${SAMPLING_METHOD}
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
    --train_batch_size 8 \
    --eval_batch_size 8 \
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
    --max_epochs 20 \
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
    --ensemble_type no_ensemble \
    --test_temperature 1.0 \
    --test_sample_size 8 \
    --train_temperature 2.0 \
    --train_sample_size 8 \
    --paraphrase_loss ${PARA_LOSS} \
    --sampling_method ${SAMPLING_METHOD}


rm -r -f ${model_path}/roberta_model_best_step
