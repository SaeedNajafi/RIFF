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

if [ "${CLUSTER_NAME}" = "vcluster" ]; then
    source ${VIRTUAL_ENV}/bin/activate
fi

echo "Using Python from: $(which python)"
