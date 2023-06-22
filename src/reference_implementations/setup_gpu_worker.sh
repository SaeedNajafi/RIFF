#!/bin/bash

# For reading key=value arguments
for ARGUMENT in "$@"
do
	KEY=$(echo $ARGUMENT | cut -f1 -d=)
	KEY_LENGTH=${#KEY}
	VALUE="${ARGUMENT:$KEY_LENGTH+1}"
	export "$KEY"="$VALUE"
done

echo "Hostname: $(hostname -s)"
echo "Node Rank ${SLURM_PROCID}"

if [ "${CLUSTER_NAME}" = "narval" ]; then
    # prepare environment in the narval cluster.
    module load python/3.9.6 StdEnv/2020 gcc/9.3.0 cuda/11.4 arrow/11.0.0
    source ${VIRTUAL_ENV}/bin/activate

elif [ "${CLUSTER_NAME}" = "vcluster" ]; then
    source ${VIRTUAL_ENV}/bin/activate

fi

echo "Using Python from: $(which python)"
