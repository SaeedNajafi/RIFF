#!/bin/bash

rates=(0.00001)
exps=(all_finetune)
seeds=(11 42 1993 2023 12321)
tasks=(sst2)
fewshot_sizes=(16 128)

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
                for f in ${!fewshot_sizes[@]};
                do
                    fewshot_size=${fewshot_sizes[$f]}
                    sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
                        src/reference_implementations/prompt_zoo/fewshot_lmfps.sh \
                        ./roberta-exps-logs \
                        ${exp} \
                        ${task} \
                        ${seed} \
                        2 \
                        ${fewshot_size} \
                        ${rate} \
                        0 \
                        1 \
                        0 \
                        25 \
                        mml \
                        off_policy \
                        beam_search \
                        accuracy \
                        0.6
                done
            done
        done
    done
done
