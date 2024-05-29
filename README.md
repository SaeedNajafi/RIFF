# Paraphrasing the Inputs
This repository holds all of the code associated with the research side of my projects considering prompt engineering and paraphrasing the inputs for language models.

The static code checker runs on python3.9

# Installing dependencies

## Virtualenv installing on linux VM
You can call `setup.sh` with the `OS=linux` flag. This installs python in the linux VM and installs the ML libraries for the GPU cards.
```
bash setup.sh OS=linux ENV_NAME=prompt_torch DEV=true
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

## Download data.
To download data, use the following link to get 16-shot splits for the classification tasks.

https://github.com/mingkaid/rl-prompt/tree/main/examples/few-shot-classification/data/16-shot

## Reference
Please cite the following article published in *ACL 2024 (findings)*.
```
@misc{najafi2024riff,
      title={RIFF: Learning to Rephrase Inputs for Few-shot Fine-tuning of Language Models},
      author={Saeed Najafi and Alona Fyshe},
      year={2024},
      eprint={2403.02271},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
