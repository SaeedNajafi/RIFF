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
@inproceedings{najafi-fyshe-2024-riff,
    title = "{RIFF}: Learning to Rephrase Inputs for Few-shot Fine-tuning of Language Models",
    author = "Najafi, Saeed  and
      Fyshe, Alona",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.85",
    pages = "1447--1466",
    abstract = "Pre-trained Language Models (PLMs) can be accurately fine-tuned for downstream text processing tasks. Recently, researchers have introduced several parameter-efficient fine-tuning methods that optimize input prompts or adjust a small number of model parameters (e.g LoRA). In this study, we explore the impact of altering the input text of the original task in conjunction with parameter-efficient fine-tuning methods. To most effectively rewrite the input text, we train a few-shot paraphrase model with a Maximum-Marginal Likelihood objective. Using six few-shot text classification datasets, we show that enriching data with paraphrases at train and test time enhances the performance beyond what can be achieved with parameter-efficient fine-tuning alone. The code used for our experiments can be found at https://github.com/SaeedNajafi/RIFF.",
}
```
