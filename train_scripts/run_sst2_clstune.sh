#!/bin/bash

rates=(0.01)
exps=(classifier_finetune)
seeds=(11 42 1993 2023 12321)
tasks=(sst2)
augs=(1)
fewshot_sizes=(32)

for i in ${!rates[@]};
do
    rate=${rates[$i]}
    for j in ${!exps[@]};
    do
        exp=${exps[$j]}
        for k in ${!seeds[@]};
        do
            seed=${seeds[$k]}
            for t in ${!tasks[@]};
            do
                task=${tasks[$t]}
                for g in ${!augs[@]};
                do
                    aug=${augs[$g]}
                    for f in ${!fewshot_sizes[@]};
                    do
                        fewshot_size=${fewshot_sizes[$f]}
                        bash src/reference_implementations/prompt_zoo/fewshot_data_augmentation_lambda.sh \
                            EXP_TYPE=${exp} \
                            TASK=${task} \
                            SEED=${seed} \
                            NUM_CLASSES=2 \
                            FEWSHOT_SIZE=${fewshot_size} \
                            LR=${rate} \
                            AUG=${aug} \
                            TRAIN_PARA=0 \
                            LOAD_PARA=0 \
                            LEN=25 \
                            PARA_LOSS="dummy" \
                            SAMPLING_METHOD="dummy" \
                            SAMPLING_ALG="dummy" \
                            METRIC_TO_SAVE=original_accuracy \
                            KL_COEFFICIENT=0.0
                    done
                done
            done
        done
    done
done
