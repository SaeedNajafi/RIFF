#!/bin/bash

rates=(0.00001)
exps=(all_finetune)
seeds=(42 11 1993 2023 12321)
tasks=(sst2)

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
                bash src/reference_implementations/prompt_zoo/sentiment_baselines_fewshot.sh \
                    EXP_TYPE=${exp} \
                    TASK=${task} \
                    SEED=${seed} \
                    MAIN_PATH=/home/snajafi/models-fewshot \
                    LR=${rate} \
                    AUG=1
            done
        done
    done
done
