#!/bin/bash

exps=(gradient_search)
seeds=(11 42 1993 2023 12321)
tasks=(sst2)

for j in ${!exps[@]};
do
    exp=${exps[$j]}
    for k in ${!seeds[@]};
    do
        seed=${seeds[$k]}
        for t in ${!tasks[@]};
        do
            task=${tasks[$t]}
            sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
                    src/reference_implementations/prompt_zoo/gradient_search_run_fewshot.sh \
                    ./roberta-exps-logs \
                    ${exp} \
                    ${task} \
                    ${seed} \
                    /scratch/ssd004/scratch/snajafi/data_temp/roberta-exps \
                    0.001 \
                    1
        done
    done
done
