"""This module implements different ideas for fine-tuning a backbone LM on some
downstream NLP datasets."""

import os
from abc import abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple

import numpy
import torch
from absl import flags
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, BartForConditionalGeneration, BartTokenizer, RobertaForMaskedLM, RobertaModel

from src.reference_implementations.prompt_zoo.data_utility import augment_batch, tokenize_samples
from src.reference_implementations.prompt_zoo.model_utility import (
    clear_cache,
    log_of_labels,
    mlm_log_of_labels,
    modify_inputs_outputs,
    set_random_seed,
    z_scoring,
)
from src.reference_implementations.prompt_zoo.prompt_optimizers import optimizer_definer
from src.reference_implementations.prompt_zoo.soft_prompt_modules import create_softprompt_roberta

FLAGS = flags.FLAGS

flags.DEFINE_string("pretrained_model", "roberta-large", "initial pre-trained model to use as backbone LM.")
flags.DEFINE_string("mode", "train", "the mode of run? train or test")
flags.DEFINE_string("model_path", "/tmp/", "main directory to save or load the model from")
flags.DEFINE_string("para_model_path", "/tmp/", "main directory to save or load the paraphrase model from")
flags.DEFINE_string("checkpoint", None, "checkpoint name to load from.")
flags.DEFINE_integer("top_k", 20, "Number of candidate tokens to replace the prompt token.")
flags.DEFINE_integer(
    "test_sample_size",
    8,
    "Number of paraphrases to generate top-p sampling or beam search used \
        for testing or data augmentation using the paraphraser.",
)
flags.DEFINE_integer(
    "train_sample_size",
    128,
    "Number of paraphrases to generate using top-p sampling or beam search used for training the paraphraser",
)
flags.DEFINE_integer("g_beam_size", 20, "Number of prompt templates to consider for gradient-search beam search.")
flags.DEFINE_integer("no_repeat_ngram_size", 2, "related to generation with beam size.")
flags.DEFINE_float(
    "test_temperature",
    1.0,
    "test or inference temperature for the softmax to smooth or sharpen the token probabilities.",
)
flags.DEFINE_float(
    "train_temperature", 1.5, "training temperature for the softmax to smooth or sharpen the token probabilities."
)


# details about the model
# https://huggingface.co/stanford-oval/paraphraser-bart-large.
paraphrase_model_name = "stanford-oval/paraphraser-bart-large"
flags.DEFINE_string("ensemble_type", "no_ensemble", "ensemble type with the paraphraser.")
flags.DEFINE_string("paraphrase_loss", "pg_z_score", "the training objective used to train the paraphrase model.")
flags.DEFINE_string(
    "sampling_method",
    "on_policy",
    "Whether to do on-policy sampling using the paraphrase model \
        or off-policy sampling using a separate paraphrase model.",
)
flags.DEFINE_string(
    "sampling_algorithm",
    "top_p",
    "What algorithm to use for sampling? top_p or beam_search?",
)
flags.DEFINE_float(
    "kl_penalty_coefficient",
    0.1,
    "What is the coefficient for the KL penalty used in the ppo algorithm?",
)
flags.DEFINE_integer("classifier_hidden_d", 128, "The number of hidden units used in the classifier.")
flags.DEFINE_integer("num_classes", 3, "Number of classes for classification. Only used in linear classifier.")

# path to the local pre-trained models on the narval cluster.
NARVAL_PATH = "/home/saeednjf/scratch/paraphrase_inputs_for_prompts/models"


class MyBaseLM(torch.nn.Module):
    """Base LM class for different finetuning + prompt-tuning experiments."""

    def __init__(self, seed: int, device: int) -> None:
        super().__init__()

        set_random_seed(seed)

        # check the gpu actually exists and setup device.
        self.gpu_check = torch.cuda.is_available()
        assert self.gpu_check

        self.device = torch.device(f"cuda:{device}")

        # will contain a dictionary with model name as the key
        # and the actual model as the value.
        self.model_pool: Dict[str, torch.nn.Module] = {}

        # for some subclasses, we will compute per token log probabilities.
        # pad tokens have index -100 in huggingface.
        # don't reduce loss (log likelihood), compute loss per token.
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    def setup_models(self) -> None:
        """Put models defined in GPU devices."""
        # put model on gpu.
        for model_name, model in self.model_pool.items():
            model.to(self.device)

            # compile the pytorch model for speedup.
            # compile is slow in my code!!!
            # self.model_pool[model_name] = torch.compile(model, mode="max-autotune")

        self.loss_func = self.loss_func.to(self.device)

    def load_from_checkpoint(self, model_path: Optional[str] = None) -> None:
        """Loads the weights from the given checkpoint."""
        m_path = FLAGS.model_path
        if model_path is not None:
            # set to the given model path.
            m_path = model_path
        ckp_name = FLAGS.checkpoint
        try:
            for m_name, model in self.model_pool.items():
                model_ckp = os.path.join(m_path, f"{m_name}_{ckp_name}")
                model.load_state_dict(
                    torch.load(
                        model_ckp,
                        map_location=lambda storage, loc: storage,
                    )
                )
        except Exception as e:
            raise Exception("Could not load the checkpoint due to error:{}".format(e))

    def save(self, checkpoint_name: str, model_path: Optional[str] = None) -> None:
        """Save the modules to the model_path for the specified checkpoint
        name."""
        m_path = FLAGS.model_path
        if model_path is not None:
            m_path = model_path
        if not os.path.exists(m_path):
            os.makedirs(m_path)
        for m_name, model in self.model_pool.items():
            torch.save(model.state_dict(), os.path.join(m_path, f"{m_name}_{checkpoint_name}"))

        # recursively save the inner models if present.
        if hasattr(self, "para_model"):
            if not self.para_model.fixed:
                # only save if we are training the para_model.
                self.para_model.save(checkpoint_name, FLAGS.para_model_path)

    def train_mode_on(self) -> None:
        """Before every forward-backward iteration over batch, clear gpu cache,
        turn on train mode, and zero the optimizer gradient state if
        defined!"""

        clear_cache()

        # turn on training mode which enables dropout.
        for model in self.model_pool.values():
            model.train()

        # only if the optimizer is defined.
        if hasattr(self, "optimizer"):
            self.optimizer.zero_grad()

    def predict_mode_on(self) -> None:
        """For each iteration of prediction over batch, clear gpu cache, turn
        on eval mode."""

        clear_cache()

        # turn on eval mode which disables dropout.
        for model in self.model_pool.values():
            model.eval()

    def move_to_gpu(self, batch: torch.utils.data.Dataset, keys: List[str]) -> Dict[str, torch.Tensor]:
        """If gpu flag is set, move the batch tensors specified by keys into
        the gpu and return a dictionary to access the gpu tensors."""
        return {key: batch[key].to(self.device) for key in keys}

    @abstractmethod
    def train(self, batch: torch.utils.data.Dataset) -> Dict[str, float]:
        """The abstract train function."""
        pass

    @abstractmethod
    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """The abstract predict function."""
        pass

    def bart_forward_pass(self, batch: torch.utils.data.Dataset, train: bool = False) -> torch.Tensor:
        """Run a forward computation over the batch, compute the log
        probability over the batch.

        This function can be called multiple times for training or
        inference. This function is to train the paraphraser.
        """
        if train:
            self.train_mode_on()
        else:
            self.predict_mode_on()

        loaded_batch = self.move_to_gpu(
            batch, keys=["para_input_ids", "para_attention_mask", "para_target_attention_mask", "para_labels"]
        )
        # keep an internal link to the loaded batch on gpu or cpu.
        self.loaded_batch = loaded_batch

        # we have to make sure that the PAD token is ignored.
        # huggingface ignores a pad token if the token is -100!
        orig_labels = loaded_batch["para_labels"]
        labels = orig_labels.masked_fill(orig_labels == self.tokenizer.pad_token_id, -100)

        bart_model = self.model_pool["bart_model"]

        with torch.set_grad_enabled(train):
            class_log_p = log_of_labels(
                model=bart_model,
                input_ids=loaded_batch["para_input_ids"],
                input_mask=loaded_batch["para_attention_mask"],
                decoder_mask=loaded_batch["para_target_attention_mask"],
                labels=labels,
                loss_func=self.loss_func,
            )

        return class_log_p

    def roberta_forward_pass(self, batch: torch.utils.data.Dataset, train: bool = False) -> torch.Tensor:
        """Using the Roberta Model, run a forward computation over the batch,
        compute the log probability over the batch.

        This function can be called multiple times for training or
        inference.
        """

        if train:
            self.train_mode_on()
        else:
            self.predict_mode_on()

        loaded_batch = self.move_to_gpu(batch, keys=["input_ids", "attention_mask", "input_output_ids"])

        # keep an internal link to the loaded batch on gpu or cpu.
        self.loaded_batch = loaded_batch

        # mask labels of non-<mask> tokens
        masked_labels = torch.where(
            loaded_batch["input_ids"] == self.tokenizer.mask_token_id, loaded_batch["input_output_ids"], -100
        )

        with torch.set_grad_enabled(train):
            class_log_ps = mlm_log_of_labels(
                model=self.model_pool["roberta_model"],
                input_ids=loaded_batch["input_ids"],
                input_mask=loaded_batch["attention_mask"],
                labels=masked_labels,
                loss_func=self.loss_func,
            )

        return class_log_ps

    def gradient_search_forward_pass(
        self, batch: torch.utils.data.Dataset, train: bool = False, prompt_lists: Optional[List[List[int]]] = None
    ) -> torch.Tensor:
        """Run a forward computation over the batch compute the log probability
        over the batch This function can be called multiple times for training
        or inference."""

        loaded_batch = self.move_to_gpu(batch, keys=["input_ids", "attention_mask", "input_output_ids"])
        # keep an internal link to the loaded batch on gpu or cpu.
        self.loaded_batch = loaded_batch

        # mask labels of non-<mask> tokens
        masked_labels = torch.where(
            loaded_batch["input_ids"] == self.tokenizer.mask_token_id,
            loaded_batch["input_output_ids"],
            -100,
        )
        loaded_batch["masked_labels"] = masked_labels

        if train:
            self.train_mode_on()
        else:
            self.predict_mode_on()

        modify_inputs_outputs(loaded_batch, prompt_lists)

        with torch.set_grad_enabled(train):
            class_log_ps = mlm_log_of_labels(
                model=self.model_pool["roberta_model"],
                input_ids=loaded_batch["modified_input_ids"],
                input_mask=loaded_batch["modified_attention_mask"],
                labels=loaded_batch["modified_masked_labels"],
                loss_func=self.loss_func,
            )
        return class_log_ps


class FFClassifier(torch.nn.Module):
    """A feedforward multinomial logistic regression over the LM hidden
    states."""

    def __init__(self, model_d: int) -> None:
        """Arguments:
        model_d (int): The hidden dimension of LM;
        """
        super().__init__()

        self.layer = torch.nn.Linear(model_d, FLAGS.classifier_hidden_d, bias=True)

        # using gelu activation over relu
        # https://arxiv.org/abs/1606.08415v4
        self.act = torch.nn.GELU()
        self.classifier = torch.nn.Linear(FLAGS.classifier_hidden_d, FLAGS.num_classes, bias=True)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.loss_fun = torch.nn.NLLLoss(reduction="none")

    def forward(self, hidden_states: torch.Tensor, input_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pass the hidden_vector into the classifier."""
        # mask the correct hidden_states from non-masked tokens.
        # masked tokens are zero!
        b_sz, seq_len, h_dim = hidden_states.size()
        extended_mask = input_mask.view(b_sz, seq_len, 1).expand_as(hidden_states)
        good_hidden_states = hidden_states * extended_mask
        # average pooling as the input feature vector.
        hidden_vector = torch.sum(good_hidden_states, dim=1) / torch.sum(extended_mask, dim=1)

        feature_vector = self.act(self.layer(hidden_vector))
        scores = self.classifier(feature_vector)
        logits = self.log_softmax(scores)
        return scores, logits

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        input_mask: torch.Tensor,
        class_indices: torch.Tensor,
        enable_data_augmentation: Optional[bool] = False,
    ) -> torch.Tensor:
        """Compute the cross-entropy loss for the above classifier."""
        _, logits = self.forward(hidden_states, input_mask)
        neg_log_likelihoods = self.loss_fun(logits, class_indices)
        if not enable_data_augmentation:
            return neg_log_likelihoods.mean(dim=0)
        else:
            neg_log_likelihoods = self.loss_fun(logits, class_indices)
            batch_size = class_indices.size()[0] // (FLAGS.test_sample_size + 1)
            loss = 0.0
            for idx in range(batch_size):
                idx_neg_log_likelihoods = neg_log_likelihoods[
                    idx * (FLAGS.test_sample_size + 1) : (idx + 1) * (FLAGS.test_sample_size + 1)
                ]
                idx_loss = 0.5 * idx_neg_log_likelihoods[0] + 0.5 * torch.mean(idx_neg_log_likelihoods[1:], dim=0)
                loss += idx_loss
            return loss / float(batch_size)


class Paraphraser(MyBaseLM):
    """Wrapper class around the MyBaseLM Model to load a paraphraser."""

    def __init__(self, seed: int, device: int, mode: str, fixed: bool = False) -> None:
        super().__init__(seed, device)

        # construct tokenizer
        try:
            self.tokenizer = BartTokenizer.from_pretrained(paraphrase_model_name)
            self.model_pool["bart_model"] = BartForConditionalGeneration.from_pretrained(paraphrase_model_name)
        except Exception:
            # for narval cluster.
            path = f"{NARVAL_PATH}/stanford-oval/paraphraser-bart-large"
            self.tokenizer = BartTokenizer.from_pretrained(path)
            self.model_pool["bart_model"] = BartForConditionalGeneration.from_pretrained(path)
        self.fixed = fixed

        self.setup_models()

        if not self.fixed and mode == "train":
            # to train the paraphraser, we update all of its parameters.
            temp_learning_rate = FLAGS.learning_rate
            # for paraphraser, we use the all_finetune 0.00001 learning rate.
            FLAGS.learning_rate = 0.00001
            self.optimizer = optimizer_definer["all_finetune"](self.model_pool)
            FLAGS.learning_rate = temp_learning_rate

        elif not self.fixed and mode in ["test", "inference", "eval"]:
            # load from the given checkpoint.
            self.load_from_checkpoint(FLAGS.para_model_path)

    def generate_beam_paraphrases(self, batch: torch.utils.data.Dataset, num_return_seq: int) -> List[str]:
        """The main prediction loop to generate paraphrases."""
        self.predict_mode_on()
        loaded_batch = self.move_to_gpu(batch, keys=["para_input_ids", "para_attention_mask"])

        bart_model = self.model_pool["bart_model"]

        predictions = []
        predictions_output = bart_model.generate(
            input_ids=loaded_batch["para_input_ids"],
            attention_mask=loaded_batch["para_attention_mask"],
            no_repeat_ngram_size=FLAGS.no_repeat_ngram_size,
            num_beams=num_return_seq,
            early_stopping=True,
            max_length=128,
            num_return_sequences=num_return_seq,
            output_scores=True,
            length_penalty=1.0,  # this sth that I should investigate in the future!
            return_dict_in_generate=True,
        )
        predictions.extend(predictions_output.sequences)

        # all special tokens will be removed.
        predictions_str = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        predictions_str = [pred.strip() for pred in predictions_str]
        return predictions_str

    def generate_top_p_paraphrases(
        self, batch: torch.utils.data.Dataset, num_return_seq: int, temperature: float
    ) -> List[str]:
        """The main prediction loop to generate paraphrases."""
        # This function is to provide random samples for learning with RL!
        self.predict_mode_on()
        loaded_batch = self.move_to_gpu(batch, keys=["para_input_ids", "para_attention_mask"])

        bart_model = self.model_pool["bart_model"]
        predictions_output = bart_model.generate(
            input_ids=loaded_batch["para_input_ids"],
            attention_mask=loaded_batch["para_attention_mask"],
            no_repeat_ngram_size=FLAGS.no_repeat_ngram_size,
            do_sample=True,
            top_p=0.99,
            temperature=temperature,
            max_length=128,
            num_return_sequences=num_return_seq,
            output_scores=True,
            return_dict_in_generate=True,
        )

        # all special tokens will be removed.
        predictions_str = self.tokenizer.batch_decode(predictions_output.sequences, skip_special_tokens=True)
        predictions_str = [pred.strip() for pred in predictions_str]
        return predictions_str


class RobertaPrompted(MyBaseLM):
    """Wrapper class around the Roberta-large Model to experiment with
    different finetuning or prompting ideas."""

    def __init__(
        self, seed: int, enable_data_augmentation: int, enable_paraphrase_training: int, load_paraphraser: int
    ) -> None:
        super().__init__(seed, device=0)

        try:
            # construct tokenizer.
            self.tokenizer = AutoTokenizer.from_pretrained(FLAGS.pretrained_model)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(f"{NARVAL_PATH}/roberta-large-tokenizer")

        # construct the underlying model
        if FLAGS.exp_type == "soft_prompt_finetune":
            self.model_pool["roberta_model"] = create_softprompt_roberta()

        elif FLAGS.exp_type == "classifier_finetune":
            try:
                self.model_pool["roberta_model"] = RobertaModel.from_pretrained(FLAGS.pretrained_model)
            except Exception:
                path = f"{NARVAL_PATH}/roberta-large-model"
                self.model_pool["roberta_model"] = RobertaModel.from_pretrained(path)

            # use the d_model from the LM config defined internally from huggingface.
            self.model_pool["classifier_model"] = FFClassifier(self.model_pool["roberta_model"].config.hidden_size)
        else:
            try:
                # construct the underlying model.
                self.model_pool["roberta_model"] = RobertaForMaskedLM.from_pretrained(FLAGS.pretrained_model)
            except Exception:
                path = f"{NARVAL_PATH}/roberta-large-masked-lm"
                self.model_pool["roberta_model"] = RobertaForMaskedLM.from_pretrained(path)

            if FLAGS.exp_type == "lora_finetune":
                inference_mode = False if FLAGS.mode == "train" else True
                peft_config = LoraConfig(
                    task_type=TaskType.SEQ_CLS, inference_mode=inference_mode, r=8, lora_alpha=32, lora_dropout=0.1
                )
                self.model_pool["roberta_model"] = get_peft_model(self.model_pool["roberta_model"], peft_config)
                self.model_pool["roberta_model"].print_trainable_parameters()

        self.enable_data_augmentation = enable_data_augmentation
        self.enable_paraphrase_training = enable_paraphrase_training
        if self.enable_data_augmentation == 1:
            if load_paraphraser == 1:
                # this is to load the fine-tuned paraphraser.
                self.para_model = Paraphraser(seed, device=0, mode="test", fixed=False)
                self.para_tokenizer = self.para_model.tokenizer
            else:
                # this is just to use the basic pre-trained paraphraser.
                self.para_model = Paraphraser(seed, device=0, mode=FLAGS.mode, fixed=True)
                self.para_tokenizer = self.para_model.tokenizer

            self.sample_memory: Dict[str, List[str]] = {}

        elif self.enable_paraphrase_training == 1:
            if FLAGS.sampling_method in ["off_policy", "ppo"]:
                # two bart models, move to another gpu.
                self.fixed_para_model = Paraphraser(seed, device=0, mode=FLAGS.mode, fixed=True)
                self.para_model = Paraphraser(seed, device=0, mode=FLAGS.mode, fixed=False)
            elif FLAGS.sampling_method == "on_policy":
                self.para_model = Paraphraser(seed, device=0, mode=FLAGS.mode, fixed=False)
            self.para_tokenizer = self.para_model.tokenizer

        self.setup_models()

        if FLAGS.mode == "train" and FLAGS.exp_type not in ["gradient_search", "grips"]:
            # create optimizer only for training.
            # based on the experiment type, setup the optimizer.
            self.optimizer = optimizer_definer[FLAGS.exp_type](self.model_pool)

        elif FLAGS.mode in ["test", "inference", "eval"] and FLAGS.exp_type not in ["gradient_search", "grips"]:
            # load from the given checkpoint.
            self.load_from_checkpoint()

        if FLAGS.mode == "train" and self.enable_paraphrase_training == 1:
            # for training with the paraphraser, we need average ensembling prediction
            # while evaluating the checkpoints on the dev data.
            FLAGS.ensemble_type = "paraphrase_predict"

        self.grad_scalar = torch.cuda.amp.GradScaler()

    def draw_samples_for_augmentation(self, batch: torch.utils.data.Dataset) -> List[str]:
        """Draw new samples if they are not in the sample memory.

        Keep using the previous samples drawn for the previous epochs
        during the data augmentation phase.
        """
        paraphrases_input_text = self.para_tokenizer.batch_decode(batch["para_input_ids"], skip_special_tokens=True)
        batch_size = len(paraphrases_input_text)
        paraphrases_indices: Dict[int, List[str]] = {}
        missed_indices = []
        for idx, para_input_text in enumerate(paraphrases_input_text):
            if para_input_text in self.sample_memory:
                paraphrases_indices[idx] = self.sample_memory[para_input_text]
            else:
                missed_indices.append(idx)
        if len(missed_indices) > 0:
            new_paraphrases = self.para_model.generate_top_p_paraphrases(
                batch, num_return_seq=FLAGS.test_sample_size, temperature=FLAGS.test_temperature
            )
            for missed_idx in missed_indices:
                new_samples = new_paraphrases[
                    missed_idx * FLAGS.test_sample_size : (missed_idx + 1) * FLAGS.test_sample_size
                ]
                paraphrases_indices[missed_idx] = new_samples
                self.sample_memory[paraphrases_input_text[missed_idx]] = new_samples

        paraphrases = []
        for idx in range(batch_size):
            paraphrases.extend(paraphrases_indices[idx])

        return paraphrases

    def train(self, batch: torch.utils.data.Dataset) -> Dict[str, float]:
        """The main train loop for generating the class sequence in the
        backbone LM roberta-large."""

        with torch.autocast(device_type="cuda"):
            to_train_lm = True
            if self.enable_data_augmentation == 1:
                potentials_str = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                paraphrases = self.draw_samples_for_augmentation(batch)
                augment_batch(
                    batch, paraphrases, self.tokenizer, potentials_str, num_return_seq=FLAGS.test_sample_size
                )
                to_train_lm = True

            elif self.enable_paraphrase_training == 1:
                potentials_str = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

                if FLAGS.sampling_method in ["on_policy", "ppo"]:
                    sampling_model = self.para_model

                elif FLAGS.sampling_method == "off_policy":
                    sampling_model = self.fixed_para_model

                if FLAGS.sampling_algorithm == "top_p":
                    samples = sampling_model.generate_top_p_paraphrases(
                        batch, num_return_seq=FLAGS.train_sample_size, temperature=FLAGS.train_temperature
                    )

                elif FLAGS.sampling_algorithm == "beam_search":
                    samples = sampling_model.generate_beam_paraphrases(batch, num_return_seq=FLAGS.train_sample_size)

                elif FLAGS.sampling_algorithm == "mixed":
                    top_p_samples = sampling_model.generate_top_p_paraphrases(
                        batch, num_return_seq=FLAGS.train_sample_size, temperature=FLAGS.train_temperature
                    )
                    beam_samples = sampling_model.generate_beam_paraphrases(
                        batch, num_return_seq=FLAGS.train_sample_size
                    )
                    batch_size = len(top_p_samples) // FLAGS.train_sample_size

                    # the array to hold the mixed samples from beam-search and top-p sampling.
                    samples = []
                    for idx in range(batch_size):
                        top_p_sample_arr = top_p_samples[
                            idx * FLAGS.train_sample_size : (idx + 1) * FLAGS.train_sample_size
                        ]
                        beam_sample_arr = beam_samples[
                            idx * FLAGS.train_sample_size : (idx + 1) * FLAGS.train_sample_size
                        ]
                        samples.extend(top_p_sample_arr[: FLAGS.train_sample_size // 2])
                        samples.extend(beam_sample_arr[: FLAGS.train_sample_size // 2])

                batch_size, seq_len = batch["para_input_ids"].size()
                batch["para_input_ids"] = (
                    batch["para_input_ids"]
                    .reshape(batch_size, 1, seq_len)
                    .expand(batch_size, FLAGS.train_sample_size, seq_len)
                    .reshape(-1, seq_len)
                )
                batch["para_attention_mask"] = (
                    batch["para_attention_mask"]
                    .reshape(batch_size, 1, seq_len)
                    .expand(batch_size, FLAGS.train_sample_size, seq_len)
                    .reshape(-1, seq_len)
                )
                tokenize_samples(batch, samples, self.para_tokenizer)

                para_log_ps = self.para_model.bart_forward_pass(batch, train=True)

                if FLAGS.sampling_method in ["off_policy", "ppo"]:
                    # we also need this for off-policy sampling.
                    fixed_para_log_ps = self.fixed_para_model.bart_forward_pass(batch, train=False)

                augment_batch(batch, samples, self.tokenizer, potentials_str, num_return_seq=FLAGS.train_sample_size)
                to_train_lm = False

            if FLAGS.exp_type == "soft_prompt_finetune":
                class_log_ps = self.gradient_search_forward_pass(batch, train=to_train_lm, prompt_lists=None)
            else:
                class_log_ps = self.roberta_forward_pass(batch, train=to_train_lm)

            if self.enable_paraphrase_training == 1:
                class_log_ps = class_log_ps.reshape(batch_size, FLAGS.train_sample_size + 1)
                normal_class_log_ps = (
                    class_log_ps[:, 0].reshape(batch_size, 1).expand(batch_size, FLAGS.train_sample_size)
                )
                paraphrase_class_log_ps = class_log_ps[:, 1:]

                final_rewards_z = z_scoring(paraphrase_class_log_ps - normal_class_log_ps)

                if FLAGS.sampling_method in ["off_policy", "ppo"]:
                    para_log_ps = para_log_ps.to(self.device)
                    para_log_ps_copy = para_log_ps.detach().clone()
                    fixed_para_log_ps = fixed_para_log_ps.to(self.device)
                    importance_ratio_log = para_log_ps_copy - fixed_para_log_ps
                    importance_ratio_log = importance_ratio_log.reshape(batch_size, FLAGS.train_sample_size)
                    fixed_para_log_ps = fixed_para_log_ps.reshape(batch_size, FLAGS.train_sample_size)
                    importance_ratio = torch.exp(importance_ratio_log)
                    para_log_ps = para_log_ps.reshape(batch_size, FLAGS.train_sample_size)

                elif FLAGS.sampling_method == "on_policy":
                    para_log_ps = para_log_ps.to(self.device)
                    para_log_ps = para_log_ps.reshape(batch_size, FLAGS.train_sample_size)
                    importance_ratio = torch.ones(
                        batch_size, FLAGS.train_sample_size, device=self.device, dtype=para_log_ps.dtype
                    )

                if FLAGS.paraphrase_loss == "pg":
                    pg_loss = -torch.mean(torch.mean(importance_ratio * para_log_ps * final_rewards_z, dim=1), dim=0)
                    loss = pg_loss

                if FLAGS.paraphrase_loss == "mml":
                    if FLAGS.sampling_method == "off_policy":
                        ratio_log = para_log_ps - fixed_para_log_ps + final_rewards_z
                    elif FLAGS.sampling_method in ["on_policy", "ppo"]:
                        ratio_log = para_log_ps + final_rewards_z
                    mml_loss = -torch.mean(torch.logsumexp(ratio_log, dim=1), dim=0)
                    loss = mml_loss

                if FLAGS.sampling_method == "ppo":
                    # now we need to add the kl penalty.
                    kl_penalty = torch.mean(torch.mean((importance_ratio_log + 1) * para_log_ps, dim=1), dim=0)
                    loss = loss + FLAGS.kl_penalty_coefficient * kl_penalty

            elif self.enable_data_augmentation == 1:
                class_log_ps = class_log_ps.reshape(-1, FLAGS.test_sample_size + 1)
                normal_class_log_ps = class_log_ps[:, 0]
                paraphrase_class_log_ps = class_log_ps[:, 1:]
                objective = 0.5 * normal_class_log_ps + 0.5 * paraphrase_class_log_ps.mean(dim=1)
                loss = -objective.mean(dim=0)

            else:
                # average log probs over the batch dimension.
                loss = -class_log_ps.mean(dim=0)

        self.grad_scalar.scale(loss).backward()
        if self.enable_paraphrase_training == 1:
            # updates only the paraphrase model.
            self.grad_scalar.step(self.para_model.optimizer)
            self.grad_scalar.update()
        else:
            # updates the main language model.
            self.grad_scalar.step(self.optimizer)

        self.grad_scalar.update()
        loss_value = loss.item()

        return {"loss_value": loss_value}

    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """Based on the ensembling type, run the prediction."""
        if FLAGS.ensemble_type == "paraphrase_predict":
            return self.paraphrase_and_predict(batch)
        return self.no_ensemble_predict(batch)

    def no_ensemble_predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """The main prediction loop for a given potential class verbalizer."""

        inputs_str = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
        potentials_str = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        if FLAGS.exp_type == "soft_prompt_finetune":
            class_log_ps = self.gradient_search_forward_pass(batch, train=False, prompt_lists=None)
        else:
            class_log_ps = self.roberta_forward_pass(batch, train=False)

        class_log_ps = class_log_ps.cpu().detach().numpy()
        for index, potential_str in enumerate(potentials_str):
            output_row = {
                "potential_class": potential_str.strip(),
                "original_prediction_score": class_log_ps[index],
                "original_inputs": inputs_str[index].strip(),
                "gold_class": batch["gold_classes"][index],
            }
            yield output_row

    def paraphrase_and_predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """The main prediction loop for a given potential class verbalizer
        Generate multiple paraphrases along the original input and return all
        the results.

        For each example, the first score belongs to the original input.
        """

        potentials_str = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        inputs_str = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
        paraphrases = self.para_model.generate_top_p_paraphrases(
            batch, num_return_seq=FLAGS.test_sample_size, temperature=FLAGS.test_temperature
        )
        augment_batch(batch, paraphrases, self.tokenizer, potentials_str, num_return_seq=FLAGS.test_sample_size)

        if FLAGS.exp_type == "soft_prompt_finetune":
            class_log_ps = self.gradient_search_forward_pass(batch, train=False, prompt_lists=None)
        else:
            class_log_ps = self.roberta_forward_pass(batch, train=False)

        class_log_ps = class_log_ps.cpu().detach().numpy()

        for index, potential_str in enumerate(potentials_str):
            scores = class_log_ps[index * (FLAGS.test_sample_size + 1) : (index + 1) * (FLAGS.test_sample_size + 1)]
            avg_score = numpy.mean(scores[1:])
            para_index = (index // FLAGS.num_classes) * FLAGS.num_classes
            output_row = {
                "potential_class": potential_str.strip(),
                "prediction_score": avg_score,
                "all_prediction_score": 0.5 * scores[0] + 0.5 * avg_score,
                "original_prediction_score": scores[0],
                "gold_class": batch["gold_classes"][index],
                "paraphrases": paraphrases[
                    para_index * FLAGS.test_sample_size : (para_index + 1) * FLAGS.test_sample_size
                ],
                "original_inputs": inputs_str[index].strip(),
            }
            yield output_row
