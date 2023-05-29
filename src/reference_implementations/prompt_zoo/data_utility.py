"""This module implements the functions for preprocessing the data files into
pytorch datasets."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from absl import flags
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, T5Tokenizer

FLAGS = flags.FLAGS
flags.DEFINE_integer("train_batch_size", 16, "The batch size used for training.")
flags.DEFINE_integer("eval_batch_size", 2048, "The batch size used for inference on the test or validation data.")
flags.DEFINE_integer("source_max_length", 128, "The maximum number of tokens consider in the input sequence.")
flags.DEFINE_integer("decoder_max_length", 128, "The maximum number of tokens consider in the output sequence.")
flags.DEFINE_string(
    "classification_type",
    "fewshot",
    "If fewshot, pick 16 examples per labels, pick the same number of train examples for the development set.",
)
flags.DEFINE_integer("fewshot_sample_size", 16, "size of samples to take for fewshot learning per class.")
flags.DEFINE_integer("seed", 42, "the seed number")
flags.DEFINE_string("instruction_type", "qa", "The intruction type to format the input sentences.")


@dataclass
class SentimentRawData:
    """Input/Output/Classes for sentiment classification raw data."""

    inputs: List[str]
    outputs: List[str]
    class_indices: List[int]
    input_outputs: List[str]
    gold_outputs: List[str]
    paraphrase_inputs: List[str]


def white_space_fix(text: str) -> str:
    """Remove extra spaces in text."""
    return " ".join(text.split())


def return_instruction() -> str:
    """Return the instruction type."""
    instruction = ""
    if FLAGS.instruction_type == "manual_template_research_sst2_with_instruction":
        instruction = "In this task, you are given sentences from movie reviews. \
            The task is to classify a sentence as 'great' if the sentiment of the \
            sentence is positive or as 'terrible' if the sentiment of the sentence is negative."
    elif FLAGS.instruction_type == "manual_template_research_sst5_with_instruction":
        instruction = "In this task, you are given sentences from movie reviews. \
            Based on the given review, classify it to one of the five classes: terrible, bad, okay, good, and great."
    return instruction


def tokenize_samples(batch: torch.utils.data.Dataset, samples: List[str], tokenizer: AutoTokenizer) -> None:
    samples = [white_space_fix(f"<s> {sample} </s>") for sample in samples]
    output_encodings = tokenizer(
        samples,
        truncation=True,
        padding="max_length",
        max_length=FLAGS.source_max_length,
        add_special_tokens=False,
    )
    batch["para_target_attention_mask"] = torch.tensor(output_encodings.attention_mask)
    batch["para_labels"] = torch.tensor(output_encodings.input_ids)


def augment_batch(
    batch: torch.utils.data.Dataset,
    paraphrases: List[str],
    tokenizer: AutoTokenizer,
    labels: List[str],
    num_return_seq: int,
    for_gradient_search: Optional[bool] = False,
) -> None:
    """augment the batch with paraphrases."""
    batch_size, seq_len = batch["input_ids"].size()
    input_ids = batch.pop("input_ids").reshape(batch_size, 1, seq_len)
    attention_mask = batch.pop("attention_mask").reshape(batch_size, 1, seq_len)
    input_output_ids = batch.pop("input_output_ids").reshape(batch_size, 1, seq_len)

    inputs = []
    input_outputs = []
    instruction = return_instruction()
    for index, label in enumerate(labels):
        for par_index in range(num_return_seq):
            par_base_index = (index // FLAGS.num_classes) * FLAGS.num_classes * num_return_seq
            paraphrase = paraphrases[par_base_index + par_index : par_base_index + par_index + 1][0]
            if instruction == "":
                if for_gradient_search:
                    # for gradient_search.
                    inputs.append(white_space_fix(f"<s> {paraphrase} It was <mask> . </s>"))
                    input_outputs.append(white_space_fix(f"<s> {paraphrase} It was {label} . </s>"))
                else:
                    # for classifier-finetuning.
                    inputs.append(white_space_fix(f"<s> {paraphrase} </s>"))
                    input_outputs.append(white_space_fix(f"<s> {paraphrase} </s>"))
            else:
                inputs.append(white_space_fix(f"<s> {instruction} {paraphrase} It was <mask> . </s>"))
                input_outputs.append(white_space_fix(f"<s> {instruction} {paraphrase} It was {label} . </s>"))

    input_encodings = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=FLAGS.source_max_length,
        add_special_tokens=False,
    )
    input_output_encodings = tokenizer(
        input_outputs,
        truncation=True,
        padding="max_length",
        max_length=FLAGS.source_max_length,
        add_special_tokens=False,
    )

    par_input_ids = torch.tensor(input_encodings.input_ids).reshape(batch_size, num_return_seq, seq_len)
    par_attention_mask = torch.tensor(input_encodings.attention_mask).reshape(batch_size, num_return_seq, seq_len)
    par_input_output_ids = torch.tensor(input_output_encodings.input_ids).reshape(batch_size, num_return_seq, seq_len)

    batch["input_ids"] = torch.cat((input_ids, par_input_ids), dim=1).reshape(-1, seq_len)
    batch["attention_mask"] = torch.cat((attention_mask, par_attention_mask), dim=1).reshape(-1, seq_len)
    batch["input_output_ids"] = torch.cat((input_output_ids, par_input_output_ids), dim=1).reshape(-1, seq_len)


def template_data(
    class_to_id: Dict[str, int],
    sentences: List[str],
    labels: List[str],
    repeat_input: bool,
) -> SentimentRawData:
    """Helper function to format the data for the models.

    if with_instructions is True, we will add an instruction to the input sentence
    and make the input a template with special keywords "instructions:", "sentence:", and "sentiment:".

    if the repeat_input is True, we will repeat the input multiple times for every possible output class.

    Finally, the end of sentence token </s> used with T5 models are added to both input and output.
    """
    paraphrase_sentences = [f"{sent}" for sent in sentences]
    if FLAGS.instruction_type == "qa":
        instruction = "what would be the sentiment of the sentence?"
        sentences = [f"question: {instruction} context: {sent}" for sent in sentences]
    elif FLAGS.instruction_type == "instruction_at_start":
        instruction = "Generate the sentiment of the next sentence."
        sentences = [f"{instruction} {sent}" for sent in sentences]
    elif FLAGS.instruction_type == "no_instruction":
        sentences = sentences
    elif FLAGS.instruction_type == "instruction_at_end":
        instruction = "Generate the sentiment of the previous sentence."
        sentences = [f"{sent} {instruction}" for sent in sentences]
    elif FLAGS.instruction_type == "manual_template_research_sst2_with_instruction":
        instruction = "In this task, you are given sentences from movie reviews. \
            The task is to classify a sentence as 'great' if the sentiment of the \
                sentence is positive or as 'terrible' if the sentiment of the sentence is negative."
        sentences = [f"{instruction} {sent} It was <mask> ." for sent in sentences]
    elif FLAGS.instruction_type == "manual_template_research_sst5_with_instruction":
        instruction = "In this task, you are given sentences from movie reviews. \
            Based on the given review, classify it to one of the five classes: terrible, bad, okay, good, and great."
        sentences = [f"{instruction} {sent} It was <mask> ." for sent in sentences]
    elif FLAGS.instruction_type in [
        "manual_template_research_sst2_no_instruction",
        "manual_template_research_sst5_no_instruction",
    ]:
        sentences = [f"{sent} It was <mask> ." for sent in sentences]

    if repeat_input:
        # repeat every input for every possible output class.
        # the inference will compute the score for every possible
        # label and then select the label with the max score given by the LM.
        inputs = []
        outputs = []
        gold_outputs = []
        input_outputs = []
        class_indices = []
        paraphrase_inputs = []
        for idx, sent in enumerate(sentences):
            for label in class_to_id.keys():
                inputs.append(white_space_fix(f"<s> {sent} </s>"))
                paraphrase_inputs.append(white_space_fix(f"<s> {paraphrase_sentences[idx]} </s>"))
                outputs.append(white_space_fix(f"<s> {label} </s>"))
                gold_outputs.append(white_space_fix(labels[idx]))
                input_output = sent.replace("<mask>", label)
                input_outputs.append(white_space_fix(f"<s> {input_output} </s>"))
                class_indices.append(class_to_id[label])
        return SentimentRawData(
            inputs=inputs,
            outputs=outputs,
            class_indices=class_indices,
            input_outputs=input_outputs,
            gold_outputs=gold_outputs,
            paraphrase_inputs=paraphrase_inputs,
        )

    # add end of sequence token.
    inputs = [white_space_fix(f"<s> {sent} </s>") for sent in sentences]
    outputs = [white_space_fix(f"<s> {label} </s>") for label in labels]
    input_outputs = [white_space_fix(inputs[i].replace("<mask>", labels[i])) for i in range(len(labels))]
    class_indices = [class_to_id[label] for label in labels]
    paraphrase_inputs = [white_space_fix(f"<s> {sent} </s>") for sent in paraphrase_sentences]
    return SentimentRawData(
        inputs=inputs,
        outputs=outputs,
        class_indices=class_indices,
        input_outputs=input_outputs,
        gold_outputs=labels,
        paraphrase_inputs=paraphrase_inputs,
    )


def read_sst_sentiment_file(split_name: str, task_name: str, repeat_input: bool = False) -> SentimentRawData:
    """Load the sst sentiment analysis split for train, validation or test."""
    assert split_name in {"train", "validation", "test"}
    dataset = load_dataset(task_name, split=split_name)
    sst5_mapping = {"0": "terrible", "1": "bad", "2": "okay", "3": "good", "4": "great"}
    sst2_mapping = {"0": "terrible", "1": "great"}
    sst5_class_to_id_mapping = {val: int(key) for key, val in sst5_mapping.items()}
    sst2_class_to_id_mapping = {val: int(key) for key, val in sst2_mapping.items()}
    if task_name == "sst2":
        class_to_id = sst2_class_to_id_mapping
    elif task_name == "SetFit/sst5":
        class_to_id = sst5_class_to_id_mapping

    def process_row(row: Dict[str, str]) -> Dict[str, str]:
        """Helper function to process each row of the dataset."""
        # verbalizers are from the following paper.
        # https://arxiv.org/pdf/2205.12548.pdf.
        if task_name == "sst2":
            label = sst2_mapping[str(row["label"])]
            return {"sentence": white_space_fix(row["sentence"]), "sentiment": label}
        elif task_name == "SetFit/sst5":
            label = sst5_mapping[str(row["label"])]
            return {"sentence": white_space_fix(row["text"]), "sentiment": label}
        return {"sentence": "none"}

    if task_name == "sst2":
        new_dataset = dataset.map(
            process_row,
            remove_columns=["idx", "label"],
        )
    elif task_name == "SetFit/sst5":
        new_dataset = dataset.map(
            process_row,
            remove_columns=["text", "label", "label_text"],
        )

    if FLAGS.classification_type == "fewshot" and split_name not in ["validation", "test"]:
        new_dataset = new_dataset.shuffle(FLAGS.seed)

    if task_name == "sst2":
        label_counter = {val: 0 for val in sst2_mapping.values()}
    elif task_name == "SetFit/sst5":
        label_counter = {val: 0 for val in sst5_mapping.values()}

    sentences = []
    labels = []
    for row in new_dataset:
        label_counter[row["sentiment"]] = label_counter.get(row["sentiment"], 0) + 1
        if FLAGS.classification_type == "fewshot" and split_name not in ["validation", "test"]:
            if label_counter[row["sentiment"]] <= FLAGS.fewshot_sample_size:
                sentences.append(row["sentence"])
                labels.append(row["sentiment"])
        else:
            sentences.append(row["sentence"])
            labels.append(row["sentiment"])

    return template_data(class_to_id, sentences, labels, repeat_input)


class SentimentDataset(Dataset):
    """Subclass the pytorch's Dataset to build my own dataset for the sentiment
    analysis task."""

    def __init__(self, data: Dict[str, Union[List[int], List[List[int]]]]) -> None:
        """store the reference to the tokenized data."""
        self.data = data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return the elements for example index 'idx' as a dictionary with
        tensor values."""
        ret = {}
        for key, val in self.data.items():
            if isinstance(val[idx], str):
                ret[key] = val[idx]
            else:
                ret[key] = torch.tensor(val[idx])
        return ret

    def __len__(self) -> int:
        """Return the length of the data."""
        return len(self.data["input_ids"])


def tokenize_data(
    rawdata: SentimentRawData, tokenizer: T5Tokenizer, para_tokenizer: Optional[T5Tokenizer] = None
) -> SentimentDataset:
    """Tokenize data into a dataset."""
    input_encodings = tokenizer(
        rawdata.inputs,
        truncation=True,
        padding="max_length",
        max_length=FLAGS.source_max_length,
        add_special_tokens=False,
    )
    output_encodings = tokenizer(
        rawdata.outputs,
        truncation=True,
        padding="max_length",
        max_length=FLAGS.decoder_max_length,
        add_special_tokens=False,
    )
    input_output_encodings = tokenizer(
        rawdata.input_outputs,
        truncation=True,
        padding="max_length",
        max_length=FLAGS.source_max_length,
        add_special_tokens=False,
    )

    data = {
        "input_ids": input_encodings.input_ids,
        "attention_mask": input_encodings.attention_mask,
        "labels": output_encodings.input_ids,
        "target_attention_mask": output_encodings.attention_mask,
        "class_indices": rawdata.class_indices,
        "input_output_ids": input_output_encodings.input_ids,
        "input_output_attention_mask": input_output_encodings.attention_mask,
        "gold_classes": rawdata.gold_outputs,
    }

    if para_tokenizer is not None:
        para_input_encodings = para_tokenizer(
            rawdata.paraphrase_inputs,
            truncation=True,
            padding="max_length",
            max_length=FLAGS.source_max_length,
            add_special_tokens=False,
        )
        data["para_input_ids"] = para_input_encodings.input_ids
        data["para_attention_mask"] = para_input_encodings.attention_mask

    return SentimentDataset(data)


def create_sentiment_dataset(
    tokenizer: AutoTokenizer,
    file_name: str,
    task_name: str,
    shuffle: bool,
    repeat_input: bool = False,
    para_tokenizer: Optional[T5Tokenizer] = None,
) -> DataLoader:
    """Function to create the required huggingface dataset to train the T5
    models on the sentiment analysis task."""

    if task_name in ["sst2", "SetFit/sst5"]:
        rawdata = read_sst_sentiment_file(file_name, task_name, repeat_input)
    else:
        raise Exception(f"this {task_name} is not supported!")

    dataset = tokenize_data(rawdata, tokenizer, para_tokenizer)
    if shuffle:
        # this is training phase.
        dataloader = DataLoader(dataset, batch_size=FLAGS.train_batch_size, shuffle=True)
    else:
        # this is inference phase.
        # keep repeated inputs in the same batch:
        FLAGS.eval_batch_size *= FLAGS.num_classes
        dataloader = DataLoader(dataset, batch_size=FLAGS.eval_batch_size, shuffle=False)
    return dataloader
