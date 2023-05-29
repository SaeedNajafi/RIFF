#!/bin/bash

rates=(0.1 0.3 0.5 0.01)
exps=(soft_prompt_finetune)
seeds=(11 1993 42 2023 12321)
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
                bash src/reference_implementations/prompt_zoo/sentiment_softprompt_baseline_fewshot.sh \
                    EXP_TYPE=${exp} \
                    TASK=${task} \
                    SEED=${seed} \
                    MAIN_PATH=~/models-fewshot/ \
                    LR=${rate} \
                    LEN=50
            done
        done
    done
done
