#!/bin/bash

rates=(0.001)
exps=(gradient_search)
seeds=(11 42 1993 2023 12321)
tasks=(sst2)
augs=(1)
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
                for g in ${!augs[@]};
                do
                    aug=${augs[$g]}
                    for f in ${!fewshot_sizes[@]};
                    do
                        fewshot_size=${fewshot_sizes[$f]}
                        sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
                            src/reference_implementations/prompt_zoo/fewshot_data_augmentation.sh \
                            ./roberta-exps-logs \
                            ${exp} \
                            ${task} \
                            ${seed} \
                            2 \
                            ${fewshot_size} \
                            ${rate} \
                            ${aug} \
                            0 \
                            1 \
                            25 \
                            mml \
                            off_policy \
                            beam_search \
                            original_accuracy \
                            0.6 \
                            /scratch/ssd004/scratch/snajafi/sst2-finetuned-paraphrase/sst2_2_${fewshot_size}_all_finetune_${seed}_0.00001_0_1_0_mml_off_policy_beam_search_accuracy_0.6
                    done
                done
            done
        done
    done
done
