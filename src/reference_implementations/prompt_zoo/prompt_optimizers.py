"""This module defines the required functions to setup the optimizers for each
experiment type with the LM."""

from typing import Dict, Optional

import apex
import torch
from absl import flags
from torch.optim.optimizer import Optimizer

FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rate", 0.005, "The learning rate used in the optimizer", lower_bound=0.0)
flags.DEFINE_float("weight_decay_rate", 0.0, "The weight decay rate used in the optimizer.", lower_bound=0.0)

OPTIMIZER_ARGS_TYPE = Dict[str, torch.nn.Module]


def construct_optimizer(model: torch.nn.Module, second_model: Optional[torch.nn.Module] = None) -> Optimizer:
    """Define the AdamW optimizer over the parameters."""

    params = list(model.parameters())
    if second_model is not None:
        # concatinate the second module parameters and register in the optimizer.
        params += list(second_model.parameters())

    # speedups with apex
    # https://nvidia.github.io/apex/optimizers.html
    optimizer = apex.optimizers.FusedAdam(params, lr=FLAGS.learning_rate, weight_decay=FLAGS.weight_decay_rate)

    return optimizer


def all_weights_opt(opt_args: OPTIMIZER_ARGS_TYPE) -> Optimizer:
    """Define the optimizer that fine-tunes all the weights in the model."""
    try:
        return construct_optimizer(model=opt_args["roberta_model"])
    except Exception:
        # for paraphraser
        # we use the small learning rate for paraphraser which tunes everything.
        return construct_optimizer(model=opt_args["bart_model"])


def input_embeddings_opt(opt_args: OPTIMIZER_ARGS_TYPE) -> Optimizer:
    """Define the optimizer that only fine-tunes the shared input embedding
    layer of the roberta model."""

    model: torch.nn.Module = opt_args["roberta_model"]
    for name, param in model.named_parameters():
        if name in [
            "roberta.embeddings.word_embeddings.weight",
            "roberta.embeddings.LayerNorm.weight",
            "roberta.embeddings.LayerNorm.bias",
        ]:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return construct_optimizer(model=model)


def output_embeddings_opt(opt_args: OPTIMIZER_ARGS_TYPE) -> Optimizer:
    """Define the optimizer that only fine-tunes the output LM head of the
    roberta model."""

    model: torch.nn.Module = opt_args["roberta_model"]
    for name, param in model.named_parameters():
        if name in [
            "lm_head.dense.weight",
            "lm_head.dense.bias",
            "lm_head.bias",
            "lm_head.layer_norm.weight",
            "lm_head.layer_norm.bias",
        ]:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return construct_optimizer(model=model)


def no_weights_opt(opt_args: OPTIMIZER_ARGS_TYPE) -> Optimizer:
    """Define the optimizer that does not fine-tune any weights."""

    model: torch.nn.Module = opt_args["roberta_model"]
    # don't waste time storing grad data.
    for _, param in model.named_parameters():
        param.requires_grad = False

    return construct_optimizer(model=model)


def lora_opt(opt_args: OPTIMIZER_ARGS_TYPE) -> Optimizer:
    """Define the optimizer that does not change the parameters defined by
    lora."""
    model: torch.nn.Module = opt_args["roberta_model"]
    return construct_optimizer(model=model)


def classifier_model_opt(opt_args: OPTIMIZER_ARGS_TYPE) -> Optimizer:
    """Define the optimizer that only fine-tunes the classifier on top of the
    LM for the downstream task."""

    model: torch.nn.Module = opt_args["roberta_model"]
    # don't waste time storing grad data.
    for _, param in model.named_parameters():
        param.requires_grad = False

    return construct_optimizer(model=model, second_model=opt_args["classifier_model"])


def prompt_model_opt(opt_args: OPTIMIZER_ARGS_TYPE) -> Optimizer:
    """Define the optimizer that only fine-tunes the prompt vectors on the
    downstream task."""

    model: torch.nn.Module = opt_args["roberta_model"]
    for name, param in model.named_parameters():
        if name == "roberta.embeddings.word_embeddings.prompt_embedder.weight":
            param.requires_grad = True
        else:
            param.requires_grad = False

    return construct_optimizer(model=model)


# store the functions that setup the optimizer for each experiment type.
optimizer_definer = {
    "all_finetune": all_weights_opt,
    "input_finetune": input_embeddings_opt,
    "output_finetune": output_embeddings_opt,
    "no_finetune": no_weights_opt,
    "classifier_finetune": classifier_model_opt,
    "soft_prompt_finetune": prompt_model_opt,
    "lora_finetune": lora_opt,
}
