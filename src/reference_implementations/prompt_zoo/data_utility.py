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
    elif FLAGS.instruction_type == "manual_template_research_agn_with_instruction":
        instruction = "In this task, you are given a news article. Your task is to classify the article to one out of the four topics \
            'World, 'Sports', 'Business', 'Tech' if the article's main topic is relevant to the world, sports, business, \
            and technology, correspondingly. If you are not sure about the topic, choose the closest option"
    elif FLAGS.instruction_type == "manual_template_research_yelp_p_with_instruction":
        instruction = "In this task, you are given Yelp reviews. The task is to classify a review as \"great\" if the overall sentiment \
            of the review is positive or as \"terrible\" if the overall sentiment of the review is negative."
    elif FLAGS.instruction_type == "manual_template_research_yelp_with_instruction":
        instruction = "In this task, you are given Yelp reviews. Based on the given review, classify it to one of the five classes:\
            (1) terrible, (2) bad, (3) okay, (4) good, and (5) great."
    elif FLAGS.instruction_type == "manual_template_research_mr_with_instruction":
        instruction = "In this task, you are given sentences from movie reviews. The task is to classify a sentence as \"great\" if\
            the sentiment of the sentence is positive or as \"terrible\" if the sentiment of the sentence is negative"
    elif FLAGS.instruction_type == "manual_template_research_cr_with_instruction":
        instruction = "In this task, you are given sentences from customer reviews. The task is to classify a sentence as \"great\" if\
            the sentiment of the sentence is positive or as \"terrible\" if the sentiment of the sentence is negative."
    return instruction


def augment_batch(
    batch: torch.utils.data.Dataset,
    paraphrases: List[str],
    tokenizer: AutoTokenizer,
    labels: List[str],
    num_return_seq: int,
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
            paraphrase = paraphrases[index * num_return_seq + par_index : index * num_return_seq + par_index + 1][0]
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
    paraphrase_sentences = [f"paraphrase: {sent}" for sent in sentences]
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
    elif FLAGS.instruction_type in {"manual_template_research_sst2_with_instruction", "manual_template_research_sst5_with_instruction", 
                                    "manual_template_research_mr_with_instruction", "manual_template_research_cr_with_instruction", 
                                    "manual_template_research_yelp_with_instruction", "manual_template_research_yelp_p_with_instruction"}:
        instruction = return_instruction()
        sentences = [f"{instruction} {sent} It was <mask> ." for sent in sentences]
    elif FLAGS.instruction_type == "manual_template_research_agn_with_instruction":
        instruction = return_instruction()
        sentences = [f"{instruction} <mask> News: {sent}" for sent in sentences]
    elif FLAGS.instruction_type in [
        "manual_template_research_sst2_no_instruction",
        "manual_template_research_sst5_no_instruction",
        "manual_template_research_agn_no_instruction",
        "manual_template_research_mr_no_instruction", 
        "manual_template_research_cr_no_instruction", 
        "manual_template_research_yelp_no_instruction", 
        "manual_template_research_yelp_p_no_instruction"
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
                paraphrase_inputs.append(white_space_fix(f"{paraphrase_sentences[idx]} </s>"))
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
    paraphrase_inputs = [white_space_fix(f"{sent} </s>") for sent in paraphrase_sentences]
    return SentimentRawData(
        inputs=inputs,
        outputs=outputs,
        class_indices=class_indices,
        input_outputs=input_outputs,
        gold_outputs=labels,
        paraphrase_inputs=paraphrase_inputs,
    )

def create_sentences_labels_table(task_name: str, split_name: str, mappings: dict, new_dataset: List[dict]):
    if task_name in ["sst2", "sst5", "rotten_tomatoes", "yelp_polarity", "yelp_review_full", "cr"]:
        label_column = "sentiment"
    elif task_name == "ag_news":
        label_column = "class"

    label_counter = {val: 0 for val in mappings.values()}

    sentences = []
    labels = []
    for row in new_dataset:
        label_counter[row[label_column]] = label_counter.get(row[label_column], 0) + 1
        if FLAGS.classification_type == "fewshot" and split_name not in ["validation", "test"]:
            if label_counter[row[label_column]] <= FLAGS.fewshot_sample_size:
                sentences.append(row[label_column])
                labels.append(row[label_column])
        else:
            sentences.append(row[label_column])
            labels.append(row[label_column])
    return sentences, labels

def read_dataset_file(split_name: str, task_name: str, mappings: dict, repeat_input: bool = False) -> SentimentRawData:
    if task_name == "sst2":
        assert split_name in {"train", "validation", "test"}
        sentence_column = "sentence"
        label_column = "sentiment"
        remove_columns=["idx", "label"]
    elif task_name == "SetFit/sst5":
        assert split_name in {"train", "validation", "test"}
        sentence_column = "text"
        label_column = "sentiment"
        remove_columns=["text", "label", "label_text"]
    elif task_name == "ag_news":
        assert split_name in {"train", "test"}
        sentence_column = "text"
        label_column = "class"
        remove_columns=["text", "label"]
    elif task_name in {"yelp_review_full", "yelp_polarity"}:
        assert split_name in {"train", "test"}
        sentence_column = "text"
        label_column = "label"
        remove_columns = ["text", "label"]
    elif task_name == "rotten_tomatoes":
        assert split_name in {"train", "validation", "test"}
        sentence_column = "text"
        label_column = "label"
        remove_columns = ["text", "label"]
    # elif task_name == "cr":
    #     assert split_name in {"train", "validation", "test"}
    #     sentence_column = "text"
    #     label_column = "label"
    #     remove_columns = ["text", "label"]
    dataset = load_dataset(task_name, split=split_name)
    class_to_id = {val: int(key) for key, val in mappings.items()}
    
    def process_row(row: Dict[str, str]) -> Dict[str, str]:
        """Helper function to process each row of the dataset."""
        # verbalizers are from the following paper.
        # https://arxiv.org/pdf/2205.12548.pdf.
        label = mappings[str(row["label"])]
        return {"sentence": white_space_fix(row[sentence_column]), label_column: label}
    
    new_dataset = dataset.map(
            process_row,
            remove_columns=remove_columns,
        )
    if FLAGS.classification_type == "fewshot" and split_name not in ["validation", "test"]:
        new_dataset = new_dataset.shuffle(FLAGS.seed)
    sentences, labels = create_sentences_labels_table(task_name, split_name, mappings, new_dataset)
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


def create_dataset(
    tokenizer: AutoTokenizer,
    file_name: str,
    task_name: str,
    shuffle: bool,
    repeat_input: bool = False,
    para_tokenizer: Optional[T5Tokenizer] = None,
) -> DataLoader:
    """Function to create the required huggingface dataset to train the T5
    models on the sentiment analysis or classification task."""
    task_mappings = {"sst2": {"0": "terrible", "1": "great"}, 
                     "SetFit/sst5": {"0": "terrible", "1": "bad", "2": "okay", "3": "good", "4": "great"}, 
                     "ag_news": {"0":"world", "1":"sports", "2":"business", "3":"sci/tech"},
                     "yelp": {"0":"terrible", "1":"bad", "2":"okay", "3":"good", "4":"great"},
                     "yelp_p": {"0": "terrible", "1": "great"},
                     "mr": {"0": "terrible", "1": "great"},
                     "cr": {"0": "terrible", "1": "great"}}
    if task_name in task_mappings.keys():
        rawdata = read_dataset_file(file_name, task_name, task_mappings[task_name], repeat_input)
    else:
        raise Exception(f"this {task_name} is not supported!")

    dataset = tokenize_data(rawdata, tokenizer, para_tokenizer)
    if shuffle:
        # this is training phase.
        dataloader = DataLoader(dataset, batch_size=FLAGS.train_batch_size, shuffle=True)
    else:
        # this is inference phase.
        dataloader = DataLoader(dataset, batch_size=FLAGS.eval_batch_size, shuffle=False)
    return dataloader
