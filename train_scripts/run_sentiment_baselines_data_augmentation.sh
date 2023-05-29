#!/bin/bash

rates=(0.00001)
exps=(all_finetune)
seeds=(11 42 1993 2023 12321)
tasks=(sst2)
augs=(1)

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
                    sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
                        src/reference_implementations/prompt_zoo/sst2_fewshot_data_augmentation.sh \
                        ./roberta-exps-logs \
                        ${exp} \
                        ${task} \
                        ${seed} \
                        /scratch/ssd004/scratch/snajafi/emnlp-2023-roberta-exps/128-shot \
                        ${rate} \
                        ${aug} \
                        0 \
                        0
                done
            done
        done
    done
done

rates=(0.001)
exps=(input_finetune output_finetune)
seeds=(11 42 1993 2023 12321)
tasks=(sst2)
augs=(1)

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
                    sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
                        src/reference_implementations/prompt_zoo/sst2_fewshot_data_augmentation.sh \
                        ./roberta-exps-logs \
                        ${exp} \
                        ${task} \
                        ${seed} \
                        /scratch/ssd004/scratch/snajafi/emnlp-2023-roberta-exps/128-shot \
                        ${rate} \
                        ${aug} \
                        0 \
                        0
                done
            done
        done
    done
done

:'
rates=(0.01)
exps=(classifier_finetune)
seeds=(11 42 1993 2023 12321)
tasks=(sst2)
augs=(0 1)

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
                    sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
                        src/reference_implementations/prompt_zoo/sst2_fewshot_data_augmentation.sh \
                        ./roberta-exps-logs \
                        ${exp} \
                        ${task} \
                        ${seed} \
                        /scratch/ssd004/scratch/snajafi/emnlp-2023-roberta-exps/128-shot \
                        ${rate} \
                        ${aug} \
                        0 \
                        0
                done
            done
        done
    done
done

rates=(0.001)
exps=(soft_prompt_finetune)
seeds=(11 42 1993 2023 12321)
tasks=(sst2)
augs=(0 1)

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
                    sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
                        src/reference_implementations/prompt_zoo/sst2_softprompt_baseline_fewshot.sh \
                        ./roberta-exps-logs \
                        ${exp} \
                        ${task} \
                        ${seed} \
                        /scratch/ssd004/scratch/snajafi/emnlp-2023-roberta-exps/128-shot \
                        ${rate} \
                        ${aug} \
                        0 \
                        0 \
                        25
                done
            done
        done
    done
done
'
