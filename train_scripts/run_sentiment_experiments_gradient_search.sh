#!/bin/bash

exps=(gradient_search)
seeds=(42 11 1993 2023 12321)
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
                    /scratch/ssd004/scratch/snajafi/data_temp/roberta-exps/sst2/10-epoch-runs/128-examples \
                    0.01 \
                    0 \
                    0 \
                    0 \
                    25
            done
        done
done
