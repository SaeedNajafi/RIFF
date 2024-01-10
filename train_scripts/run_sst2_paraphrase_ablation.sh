#!/bin/bash

rates=(0.00001)
exps=(all_finetune)
seeds=(11 42 1993 2023 12321)
tasks=(sst2)
#losses=(pg_basic mml_basic pg_zscore mml_zscore)
losses=(mml_basic)
#sampling_methods=(on_policy off_policy)
#sampling_methods=(ppo)
sampling_methods=(off_policy)
#sampling_algs=(top_p beam_search mixed)
sampling_algs=(beam_search)

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
                            TOKENIZERS_PARALLELISM=false bash src/reference_implementations/run_prompt.sh \
                                SCRIPT=src/reference_implementations/prompt_zoo/fewshot_lmfps.sh \
                                LOG_DIR=./roberta-exps-logs-lmfps \
                                EXP_TYPE=${exp} \
                                TASK=${task} \
                                SEED=${seed} \
                                NUM_CLASSES=2 \
                                FEWSHOT_SIZE=16 \
                                LR=${rate} \
                                AUG=0 \
                                TRAIN_PARA=1 \
                                LOAD_PARA=0 \
                                LEN=25 \
                                PARA_LOSS=${loss} \
                                SAMPLING_METHOD=${sampling_method} \
                                SAMPLING_ALG=${sampling_alg} \
                                METRIC_TO_SAVE=accuracy \
                                KL_COEFFICIENT=0.7 \
                                CLUSTER_NAME=vcluster
                        done
                    done
                done
            done
        done
    done
done
