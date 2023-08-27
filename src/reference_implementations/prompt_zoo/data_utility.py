"""This module implements the functions for preprocessing the data files into
pytorch datasets."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

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
class ClassificationRawData:
    """Input/Classes for classification raw data."""

    inputs: List[str]
    gold_outputs: List[str]
    paraphrase_inputs: List[str]


def white_space_fix(text: str) -> str:
    """Remove extra spaces in text."""
    return " ".join(text.split())


def return_class_to_id() -> Dict[str, str]:
    if "sst2" in FLAGS.instruction_type:
        return {"terrible": "0", "great": "1"}
    elif "sst5" in FLAGS.instruction_type:
        return {"terrible": "0", "bad": "1", "okay": "2", "good": "3", "great": "4"}
    elif "agn" in FLAGS.instruction_type:
        return {"world": "0", "sports": "1", "business": "2", "sci/tech": "3"}

    return {}


def return_instruction() -> str:
    """Return the instruction type."""
    instruction = ""
    if FLAGS.instruction_type == "manual_template_research_sst2_with_instruction":
        instruction = "In this task, you are given sentences from movie reviews. \
            The task is to classify a sentence as 'great' if the sentiment of the \
            sentence is positive or as 'terrible' if the sentiment of the sentence is negative."
    elif FLAGS.instruction_type == "manual_template_research_sst5_with_instruction":
        instruction = "In this task, you are given sentences from movie reviews. \
            Based on the given review, classify it to one of the five classes: \
                (1) terrible, (2) bad, (3) okay, (4) good, and (5) great."
    elif FLAGS.instruction_type == "manual_template_research_agn_with_instruction":
        instruction = "In this task, you are given a news article. Your task is to classify \
            the article to one out of the four topics 'World', 'Sports', 'Business', 'Tech' \
            if the article's main topic is relevant to the world, sports, business, \
            and technology, correspondingly. If you are not sure about the topic, choose the closest option."
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
    num_return_seq: int,
    remove_instruction: Optional[bool] = False,
) -> None:
    """Augment the batch with paraphrases."""
    batch_size, seq_len = batch["input_ids"].size()
    input_ids = batch.pop("input_ids").reshape(batch_size, 1, seq_len)
    attention_mask = batch.pop("attention_mask").reshape(batch_size, 1, seq_len)

    inputs = []
    instruction = return_instruction()
    for index in range(batch_size):
        for par_index in range(num_return_seq):
            par_base_index = index * num_return_seq
            paraphrase = paraphrases[par_base_index + par_index : par_base_index + par_index + 1][0]
            if FLAGS.instruction_type == "manual_template_research_agn_with_instruction":
                inputs.append(white_space_fix(f"<s> {instruction} <mask> News: {paraphrase} </s>"))
            else:
                inputs.append(white_space_fix(f"<s> {instruction} {paraphrase} It was <mask> . </s>"))
            if remove_instruction:
                # for gradient_search.
                inputs.append(inputs.pop().replace(f" {instruction} ", " "))

    input_encodings = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=FLAGS.source_max_length,
        add_special_tokens=False,
    )

    par_input_ids = torch.tensor(input_encodings.input_ids).reshape(batch_size, num_return_seq, seq_len)
    par_attention_mask = torch.tensor(input_encodings.attention_mask).reshape(batch_size, num_return_seq, seq_len)

    batch["input_ids"] = torch.cat((input_ids, par_input_ids), dim=1).reshape(-1, seq_len)
    batch["attention_mask"] = torch.cat((attention_mask, par_attention_mask), dim=1).reshape(-1, seq_len)


def template_data(sentences: List[str], labels: List[str]) -> ClassificationRawData:
    """Helper function to format the data for the models.

    if with_instructions is True, we will add an instruction to the
    input sentence and make the input a template with special keywords
    "instructions:", "sentence:", and "sentiment:".

    if the repeat_input is True, we will repeat the input multiple times
    for every possible output class.

    Finally, the end of sentence token </s> used with T5 models are
    added to both input and output.
    """
    paraphrase_sentences = [f"{sent}" for sent in sentences]
    instruction = return_instruction()
    if FLAGS.instruction_type in {
        "manual_template_research_sst2_with_instruction",
        "manual_template_research_sst5_with_instruction",
    }:
        sentences = [f"{instruction} {sent} It was <mask> ." for sent in sentences]
    elif FLAGS.instruction_type == "manual_template_research_agn_with_instruction":
        sentences = [f"{instruction} <mask> News: {sent}" for sent in sentences]
    elif FLAGS.instruction_type in [
        "manual_template_research_sst2_no_instruction",
        "manual_template_research_sst5_no_instruction",
    ]:
        sentences = [f"{sent} It was <mask> ." for sent in sentences]
    elif FLAGS.instruction_type == "manual_template_research_agn_no_instruction":
        sentences = [f"<mask> News: {sent}" for sent in sentences]

    inputs = [white_space_fix(f"<s> {sent} </s>") for sent in sentences]
    gold_outputs = [label for label in labels]
    paraphrase_inputs = [white_space_fix(f"<s> {sent} </s>") for sent in paraphrase_sentences]
    return ClassificationRawData(
        inputs=inputs,
        gold_outputs=gold_outputs,
        paraphrase_inputs=paraphrase_inputs,
    )


def read_sst_sentiment_file(
    class_to_id: Dict[str, str], split_name: str, task_name: str
) -> Tuple[Union[ClassificationRawData, None], Union[ClassificationRawData, None]]:
    """Load the sst sentiment analysis split for train, validation or test."""
    assert split_name in {"train", "validation", "test"}
    if task_name == "SetFit_sst5":
        dataset = load_dataset("SetFit/sst5", split=split_name)
    else:
        dataset = load_dataset(task_name, split=split_name)

    id_to_class = {id: label for label, id in class_to_id.items()}

    def process_row(row: Dict[str, str]) -> Dict[str, str]:
        """Helper function to process each row of the dataset."""
        # verbalizers are from the following paper.
        # https://arxiv.org/pdf/2205.12548.pdf.
        label = id_to_class[str(row["label"])]
        if "sentence" in row:
            return {"sentence": white_space_fix(row["sentence"]), "sentiment": label}

        return {"sentence": white_space_fix(row["text"]), "sentiment": label}

    if task_name == "sst2":
        new_dataset = dataset.map(
            process_row,
            remove_columns=["idx", "label"],
        )
    elif task_name == "SetFit_sst5":
        new_dataset = dataset.map(
            process_row,
            remove_columns=["text", "label", "label_text"],
        )
    elif task_name == "ag_news":
        new_dataset = dataset.map(
            process_row,
            remove_columns=["text", "label"],
        )

    if split_name in ["validation", "test"]:
        sentences = []
        labels = []
        for row in new_dataset:
            sentences.append(row["sentence"])
            labels.append(row["sentiment"])
        return None, template_data(sentences, labels)

    elif FLAGS.classification_type == "fewshot":
        new_dataset = new_dataset.shuffle(FLAGS.seed)
        # train for fewshot.
        label_counter = {label: 0 for label in class_to_id.keys()}
        train_sentences = []
        train_labels = []
        val_sentences = []
        val_labels = []
        for row in new_dataset:
            label_counter[row["sentiment"]] = label_counter.get(row["sentiment"], 0) + 1
            if label_counter[row["sentiment"]] <= FLAGS.fewshot_sample_size:
                train_sentences.append(row["sentence"])
                train_labels.append(row["sentiment"])
            elif (FLAGS.fewshot_sample_size + 1) <= label_counter[row["sentiment"]] <= FLAGS.fewshot_sample_size * 2:
                val_sentences.append(row["sentence"])
                val_labels.append(row["sentiment"])

        return template_data(train_sentences, train_labels), template_data(val_sentences, val_labels)

    else:
        # this is train split for fullshot.
        sentences = []
        labels = []
        for row in new_dataset:
            sentences.append(row["sentence"])
            labels.append(row["sentiment"])
        return template_data(sentences, labels), None


class ClassificationDataset(Dataset):
    """Subclass the pytorch's Dataset to build my own dataset for the text classification
    task."""

    def __init__(self, data: Dict[str, Union[List[int], List[List[int]]]]) -> None:
        """Store the reference to the tokenized data."""
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
    rawdata: ClassificationRawData, tokenizer: T5Tokenizer, para_tokenizer: Optional[T5Tokenizer] = None
) -> ClassificationDataset:
    """Tokenize data into a dataset."""
    input_encodings = tokenizer(
        rawdata.inputs,
        truncation=True,
        padding="max_length",
        max_length=FLAGS.source_max_length,
        add_special_tokens=False,
    )
    data = {
        "input_ids": input_encodings.input_ids,
        "attention_mask": input_encodings.attention_mask,
        "gold_outputs": rawdata.gold_outputs,
        "paraphrase_inputs": rawdata.paraphrase_inputs,
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

    return ClassificationDataset(data)


def create_sentiment_dataset(
    tokenizer: AutoTokenizer,
    file_name: str,
    task_name: str,
    para_tokenizer: Optional[T5Tokenizer] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Function to create the required huggingface dataset to train the T5
    models on the sentiment analysis task."""

    # create a class to id in the tokenizer.
    class_to_id = return_class_to_id()
    tokenizer.class_to_id = class_to_id
    tokenizer.id_to_class = {id: label for label, id in class_to_id.items()}

    if task_name in ["sst2", "SetFit_sst5", "ag_news"]:
        train_rawdata, eval_rawdata = read_sst_sentiment_file(class_to_id, file_name, task_name)
    else:
        raise Exception(f"this {task_name} is not supported!")

    train_dataloader = None
    eval_dataloader = None
    if train_rawdata is not None:
        train_dataset = tokenize_data(train_rawdata, tokenizer, para_tokenizer)
        # this is training phase.
        train_dataloader = DataLoader(
            train_dataset, batch_size=FLAGS.train_batch_size, shuffle=True, pin_memory=True, num_workers=3
        )
    if eval_rawdata is not None:
        eval_dataset = tokenize_data(eval_rawdata, tokenizer, para_tokenizer)
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=FLAGS.eval_batch_size, shuffle=False, pin_memory=True, num_workers=3
        )

    return train_dataloader, eval_dataloader
