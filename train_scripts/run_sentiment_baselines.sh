#!/bin/bash

:'
rates=(0.00001)
exps=(all_finetune)
seeds=(11 42 1993 2023 12321)
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
                sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
                    src/reference_implementations/prompt_zoo/sentiment_baselines_fewshot.sh \
                    ./roberta-exps-logs \
                    ${exp} \
                    ${task} \
                    ${seed} \
                    /scratch/ssd004/scratch/snajafi/data_temp/roberta-exps \
                    ${rate} \
                    1 \
                    0 \
                    0
            done
        done
    done
done

rates=(0.001)
exps=(input_finetune)
seeds=(11 42 1993 2023 12321)
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
                sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
                    src/reference_implementations/prompt_zoo/sentiment_baselines_fewshot.sh \
                    ./roberta-exps-logs \
                    ${exp} \
                    ${task} \
                    ${seed} \
                    /scratch/ssd004/scratch/snajafi/data_temp/roberta-exps \
                    ${rate} \
                    1 \
                    0 \
                    0
            done
        done
    done
done


rates=(0.001)
exps=(output_finetune)
seeds=(11 42 1993 2023 12321)
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
                sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
                    src/reference_implementations/prompt_zoo/sentiment_baselines_fewshot.sh \
                    ./roberta-exps-logs \
                    ${exp} \
                    ${task} \
                    ${seed} \
                    /scratch/ssd004/scratch/snajafi/data_temp/roberta-exps \
                    ${rate} \
                    1 \
                    0 \
                    0
            done
        done
    done
done

rates=(0.001)
exps=(soft_prompt_finetune)
seeds=(11 42 1993 2023 12321)
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
                sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
                    src/reference_implementations/prompt_zoo/sentiment_softprompt_baseline_fewshot.sh \
                    ./roberta-exps-logs \
                    ${exp} \
                    ${task} \
                    ${seed} \
                    /scratch/ssd004/scratch/snajafi/data_temp/roberta-exps \
                    ${rate} \
                    1 \
                    0 \
                    0 \
                    25
            done
        done
    done
done
'

rates=(0.01)
exps=(classifier_finetune)
seeds=(11 42 1993 2023 12321)
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
                sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
                    src/reference_implementations/prompt_zoo/sentiment_baselines_fewshot.sh \
                    ./roberta-exps-logs \
                    ${exp} \
                    ${task} \
                    ${seed} \
                    /scratch/ssd004/scratch/snajafi/data_temp/roberta-exps \
                    ${rate} \
                    1 \
                    0 \
                    0
            done
        done
    done
done
