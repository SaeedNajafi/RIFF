# Paraphrasing the Inputs
This repository holds all of the code associated with the research side of my projects considering prompt engineering and paraphrasing the inputs for large language models.

The static code checker runs on python3.8

# Installing dependencies

## Virtualenv installing on Cirrus VM
You can call `setup.sh` with the `OS=cirrus` flag. This installs python in the linux VM and installs the ML libraries for the GPU cards.
```
bash setup.sh OS=cirrus ENV_NAME=prompt_torch DEV=true
```

## Virtualenv installing on Vector's GPU Cluster
You can call `setup.sh` with the `OS=vcluster` flag. This installs the required libraries to run the experiments on the vector's cluster.
```
bash setup.sh OS=vcluster ENV_NAME=prompt_torch DEV=true
```

## Using Pre-commit Hooks
To check your code at commit time
```
pre-commit install
```

You can also get pre-commit to fix your code
```
pre-commit run
```
