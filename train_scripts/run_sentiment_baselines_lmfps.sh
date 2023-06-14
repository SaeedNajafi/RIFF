#!/bin/bash

rates=(0.00001)
exps=(all_finetune)
seeds=(11 42 1993 2023 12321)
tasks=(sst2)
losses=(pg mml)

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
                        src/reference_implementations/prompt_zoo/sst2_fewshot_lmfps.sh \
                        ./roberta-exps-logs \
                        ${exp} \
                        ${task} \
                        ${seed} \
                        "dummy_main_path" \
                        ${rate} \
                        0 \
                        1 \
                        0 \
                        25 \
                        ${loss} \
                        ppo \
                        top_p \
                        accuracy \
                        0.6
                done
            done
        done
    done
done

:'
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
                    bash src/reference_implementations/prompt_zoo/sst2_fewshot_lmfps.sh \
                        EXP_TYPE=${exp} \
                        TASK=${task} \
                        SEED=${seed} \
                        MODEL_PATH="dummy_main_path" \
                        LR=${rate} \
                        AUG=0 \
                        TRAIN_PARA=1 \
                        LOAD_PARA=0 \
                        LEN=25 \
                        PARA_LOSS=${loss} \
                        SAMPLING_METHOD=off_policy \
                        SAMPLING_ALG=mixed \
                        METRIC_TO_SAVE=total_score
                done
            done
        done
    done
done
'
