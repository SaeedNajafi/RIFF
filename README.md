# RIFF
This repository holds all of the code associated with the research side of my projects considering prompt engineering and paraphrasing the inputs for language models.
This repo implements the experiments in our paper:

**RIFF: Learning to Rephrase Inputs for Few-shot Fine-tuning of Language Models**

https://arxiv.org/pdf/2403.02271

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
The static code checker runs on python3.9.

To check your code at commit time
```
pre-commit install
```

You can also get pre-commit to fix your code
```
pre-commit run
```

# Download data
To download data, use the following link to get 16-shot splits for the classification tasks.

https://github.com/mingkaid/rl-prompt/tree/main/examples/few-shot-classification/data/16-shot

# Training Scripts
To train normal models without paraphrase augmentation, use the following train script.
Currently, it will be run on a linux machine on a single GPU iteratively for different seeds, prompt optimization techniques and tasks. For a cluster of GPUs managed by **slurm**, it will submit different jobs in parallel (e.g. for the **vcluster**)
```
bash train_scripts/run_basic_experiments.sh
```

Then you can finetune the paraphraser with the RIFF objective.
```
bash train_scripts/run_paraphrase_finetuning.sh
```

Finally, fix the path for the fine-tuned paraphraser, and train the downstream LM with paraphrase augmentation.
```
bash train_scripts/run_better_experiments.sh
```

# Training Loop
The losses for the downstream language model or the paraphraser model under different optimization techniques have been implemented in the following function:
```
https://github.com/SaeedNajafi/RIFF/blob/main/src/reference_implementations/prompt_zoo/prompted_lm.py#L754
```

# Reference
If you find the paper or the code helpful, please cite the following article published in *ACL 2024 (findings)*.
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
