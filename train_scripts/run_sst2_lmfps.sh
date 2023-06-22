#!/bin/bash

rates=(0.00001)
exps=(all_finetune)
seeds=(11 42 1993 2023 12321)
tasks=(sst2)
augs=(0)
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
                        bash src/reference_implementations/prompt_zoo/fewshot_lmfps_lambda.sh \
                            EXP_TYPE=${exp} \
                            TASK=${task} \
                            SEED=${seed} \
                            NUM_CLASSES=2 \
                            FEWSHOT_SIZE=${fewshot_size} \
                            LR=${rate} \
                            AUG=${aug} \
                            TRAIN_PARA=1 \
                            LOAD_PARA=0 \
                            LEN=25 \
                            PARA_LOSS="mml" \
                            SAMPLING_METHOD="off_policy" \
                            SAMPLING_ALG="beam_search" \
                            METRIC_TO_SAVE=accuracy \
                            KL_COEFFICIENT=0.0
                    done
                done
            done
        done
    done
done
