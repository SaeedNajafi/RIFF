"""This is a module to train a classifier on top of the LM."""
from typing import Dict, Iterator, Optional, Tuple

import numpy
import torch
from absl import flags
from transformers import AutoTokenizer, RobertaModel

from src.reference_implementations.prompt_zoo.data_utility import augment_batch, tokenize_samples
from src.reference_implementations.prompt_zoo.prompt_optimizers import optimizer_definer
from src.reference_implementations.prompt_zoo.prompted_lm import MyBaseLM, Paraphraser

FLAGS = flags.FLAGS

flags.DEFINE_integer("classifier_hidden_d", 128, "The number of hidden units used in the classifier.")
flags.DEFINE_integer("num_classes", 3, "Number of classes for classification. Only used in linear classifier.")


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
        para_log_ps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the cross-entropy loss for the above classifier."""
        _, logits = self.forward(hidden_states, input_mask)
        neg_log_likelihoods = self.loss_fun(logits, class_indices)
        if para_log_ps is None:
            return neg_log_likelihoods.mean(dim=0)
        else:
            neg_log_likelihoods = self.loss_fun(logits, class_indices)
            batch_size = class_indices.size()[0] // (FLAGS.test_sample_size + 1)
            loss = 0.0
            for idx in range(batch_size):
                idx_neg_log_likelihoods = neg_log_likelihoods[
                    idx * (FLAGS.test_sample_size + 1) : (idx + 1) * (FLAGS.test_sample_size + 1)
                ]
                idx_para_log_ps = para_log_ps[idx * FLAGS.test_sample_size : (idx + 1) * FLAGS.test_sample_size]
                idx_loss = idx_neg_log_likelihoods[0] + torch.sum(
                    torch.exp(idx_para_log_ps) * idx_neg_log_likelihoods[1:], dim=0
                )
                loss += idx_loss
            return loss / float(batch_size)


class ClassifierLM(MyBaseLM):
    """Wrapper class around the LM Model with a classifier on top of the LM."""

    def __init__(self, seed: int, enable_data_augmentation: int, load_paraphraser: int) -> None:
        super().__init__(seed, device=0)

        # construct tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(FLAGS.pretrained_model)

        # construct the underlying model
        if FLAGS.exp_type == "classifier_finetune":
            self.model_pool["roberta_model"] = RobertaModel.from_pretrained(FLAGS.pretrained_model)

        # use the d_model from the LM config defined internally from huggingface.
        self.model_pool["classifier_model"] = FFClassifier(self.model_pool["roberta_model"].config.hidden_size)

        self.enable_data_augmentation = enable_data_augmentation
        if self.enable_data_augmentation == 1:
            if load_paraphraser == 1:
                # this is to load the fine-tuned paraphraser.
                self.para_model = Paraphraser(seed, device=0, mode="test", fixed=False)
                self.para_tokenizer = self.para_model.tokenizer
            else:
                # this is just to use the basic pre-trained paraphraser.
                self.para_model = Paraphraser(seed, device=0, mode=FLAGS.mode, fixed=True)
                self.para_tokenizer = self.para_model.tokenizer

        self.setup_models()

        if FLAGS.mode == "train" and FLAGS.exp_type == "classifier_finetune":
            # create optimizer only for training.
            # based on the experiment type, setup the optimizer.
            self.optimizer = optimizer_definer["classifier_finetune"](self.model_pool)

        elif FLAGS.mode in ["test", "inference", "eval"]:
            # load from the given checkpoint.
            self.load_from_checkpoint()

    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """Based on the ensembling type, run the prediction."""
        if FLAGS.ensemble_type == "paraphrase_predict":
            return self.paraphrase_and_predict(batch)
        return self.no_ensemble_predict(batch)

    def no_ensemble_predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """The main prediction loop using a separate classifier."""

        self.predict_mode_on()
        loaded_batch = self.move_to_gpu(batch, keys=["input_ids", "attention_mask"])

        encoder = self.model_pool["roberta_model"]
        classifier_model = self.model_pool["classifier_model"]

        output = encoder(
            input_ids=loaded_batch["input_ids"],
            attention_mask=loaded_batch["attention_mask"],
        )

        encoder_hidden_states = output.last_hidden_state

        _, logits = classifier_model(encoder_hidden_states, loaded_batch["attention_mask"])

        prediction_logits = logits.cpu().detach().numpy()

        inputs_str = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)
        for index, input_str in enumerate(inputs_str):
            for class_idx in range(FLAGS.num_classes):
                output_row = {
                    "potential_class": str(class_idx),
                    "prediction_score": prediction_logits[index][class_idx],
                    "original_inputs": input_str.strip(),
                    "gold_class": str(batch["class_indices"][index].item()),
                }
                yield output_row

    def paraphrase_and_predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """Generate multiple paraphrases along the original input and return all
        the results.

        For each example, the first score belongs to the original input.
        """
        self.predict_mode_on()
        # for classifier_finetuning, the dummy labels doesn't have any effect.
        dummy_labels = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        inputs_str = self.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=False)

        paraphrases = self.para_model.generate_top_p_paraphrases(
            batch, num_return_seq=FLAGS.test_sample_size, temperature=FLAGS.test_temperature
        )
        augment_batch(
            batch,
            paraphrases,
            self.tokenizer,
            dummy_labels,
            num_return_seq=FLAGS.test_sample_size,
        )

        batch_size = batch["class_indices"].size()[0]
        batch["class_indices"] = (
            batch.pop("class_indices")
            .reshape(batch_size, 1)
            .expand(batch_size, 1 + FLAGS.test_sample_size)
            .reshape(
                -1,
            )
        )

        loaded_batch = self.move_to_gpu(batch, keys=["input_ids", "attention_mask"])

        encoder = self.model_pool["roberta_model"]
        classifier_model = self.model_pool["classifier_model"]

        output = encoder(
            input_ids=loaded_batch["input_ids"],
            attention_mask=loaded_batch["attention_mask"],
        )

        encoder_hidden_states = output.last_hidden_state

        _, logits = classifier_model(encoder_hidden_states, loaded_batch["attention_mask"])

        prediction_logits = logits.cpu().detach().numpy()

        for index, input_str in enumerate(inputs_str):
            scores = prediction_logits[
                index * (FLAGS.test_sample_size + 1) : (index + 1) * (FLAGS.test_sample_size + 1), :
            ]
            for class_idx in range(FLAGS.num_classes):
                scores_str = ",".join([str(score) for score in scores[:, class_idx]])
                avg_score = numpy.mean(scores[1:, class_idx])
                output_row = {
                    "potential_class": str(class_idx),
                    "prediction_score": avg_score,
                    "all_prediction_scores": scores_str,
                    "original_prediction_score": scores[0, class_idx],
                    "gold_class": str(batch["class_indices"][index * (FLAGS.test_sample_size + 1)].item()),
                    "paraphrases": paraphrases[index * FLAGS.test_sample_size : (index + 1) * FLAGS.test_sample_size],
                    "original_inputs": input_str.strip(),
                }
                yield output_row

    def train(self, batch: torch.utils.data.Dataset) -> Dict[str, float]:
        """The classifier training step."""
        self.train_mode_on()
        para_log_ps = None
        if self.enable_data_augmentation == 1:
            # dummy_labels doesn't have any effect for classifier_finetuning.
            dummy_labels = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
            paraphrases = self.para_model.generate_top_p_paraphrases(
                batch, num_return_seq=FLAGS.test_sample_size, temperature=FLAGS.test_temperature
            )
            augment_batch(batch, paraphrases, self.tokenizer, dummy_labels, num_return_seq=FLAGS.test_sample_size)

            batch_size = batch["class_indices"].size()[0]
            batch["class_indices"] = (
                batch.pop("class_indices")
                .reshape(batch_size, 1)
                .expand(batch_size, 1 + FLAGS.test_sample_size)
                .reshape(
                    -1,
                )
            )

            # compute the log probability of the paraphrases being generated.
            batch_size, seq_len = batch["para_input_ids"].size()
            batch["para_input_ids"] = (
                batch["para_input_ids"]
                .reshape(batch_size, 1, seq_len)
                .expand(batch_size, FLAGS.test_sample_size, seq_len)
                .reshape(-1, seq_len)
            )
            batch["para_attention_mask"] = (
                batch["para_attention_mask"]
                .reshape(batch_size, 1, seq_len)
                .expand(batch_size, FLAGS.test_sample_size, seq_len)
                .reshape(-1, seq_len)
            )
            tokenize_samples(batch, paraphrases, self.para_tokenizer)
            para_log_ps = self.para_model.bart_forward_pass(batch, train=False)

        loaded_batch = self.move_to_gpu(batch, keys=["input_ids", "attention_mask", "class_indices"])

        encoder = self.model_pool["roberta_model"]
        classifier_model = self.model_pool["classifier_model"]

        output = encoder(
            input_ids=loaded_batch["input_ids"],
            attention_mask=loaded_batch["attention_mask"],
        )
        encoder_hidden_states = output.last_hidden_state
        loss = classifier_model.compute_loss(
            encoder_hidden_states,
            loaded_batch["attention_mask"],
            loaded_batch["class_indices"],
            para_log_ps=para_log_ps,
        )
        loss_value = loss.item()

        # backProp
        loss.backward()

        # optimize
        self.optimizer.step()

        return {"loss_value": loss_value}
