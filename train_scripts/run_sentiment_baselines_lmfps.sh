#!/bin/bash

rates=(0.00001)
exps=(all_finetune)
seeds=(11 42 1993 2023 12321)
tasks=(sst2)
losses=(mml_basic mml_z_score pg_basic pg_z_score pg_reward_diff)

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
                for l in ${!losses[@]};
                do
                    loss=${losses[$l]}
                    sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
                        src/reference_implementations/prompt_zoo/sst2_fewshot_lmfps_train.sh \
                        ./roberta-exps-logs \
                        ${exp} \
                        ${task} \
                        ${seed} \
                        /scratch/ssd004/scratch/snajafi/emnlp-2023-roberta-exps/128-shot \
                        ${rate} \
                        0 \
                        1 \
                        0 \
                        25 \
                        ${loss} \
                        off_policy
                done
            done
        done
    done
done
