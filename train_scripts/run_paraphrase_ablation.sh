#!/bin/bash

rates=(0.00001)
exps=(all_finetune)
seeds=(100 13 21 42 87)
num_classes=(2)
tasks=(subj)
losses=(mml_zscore)
sampling_methods=(ppo)
sampling_algs=(mixed)

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
                num_class=${num_classes[$t]}
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
                                NUM_CLASSES=${num_class} \
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
                                KL_COEFFICIENT=0.1 \
                                CLUSTER_NAME=vcluster \
                                GPU_TYPE=a40
                        done
                    done
                done
            done
        done
    done
done
