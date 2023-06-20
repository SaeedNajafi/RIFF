#!/bin/bash

rates=(0.00001)
exps=(all_finetune)
seeds=(11 42 1993 2023 12321)
tasks=(sst2)
losses=(pg mml)
sampling_methods=(on_policy off_policy)
sampling_algs=(beam_search top_p mixed)

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
                    for s_alg in ${!sampling_algs[@]};
                    do
                        sampling_alg=${sampling_algs[$s_alg]}
                        for s_m in ${!sampling_methods[@]};
                        do
                            sampling_method=${sampling_methods[$s_m]}
                            sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
                                src/reference_implementations/prompt_zoo/fewshot_lmfps.sh \
                                ./roberta-exps-logs \
                                ${exp} \
                                ${task} \
                                ${seed} \
                                2 \
                                32 \
                                ${rate} \
                                0 \
                                1 \
                                0 \
                                25 \
                                ${loss} \
                                ${sampling_method} \
                                ${sampling_alg} \
                                accuracy \
                                0.6
                        done
                    done
                done
            done
        done
    done
done
