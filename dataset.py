from datasets import load_dataset
from tokenizer import BertTokenizerForDocumentClassification
import numpy as np
from functools import partial

def multi_label_encoding(example, num_classes=2):
    label_vector = [0] * num_classes
    label_vector[example["label"]] = 1 
    example["label"] = label_vector
    return example


class BertDatasetForDocumentClassification:
    def __init__(self, dataset="ag_news", model_name="bert-base-uncased", max_length=512, split=["train", "test"]):
        self.dataset = load_dataset(dataset, split=split)[0]  # Get train dataset
        self.max_length = max_length
        self.multi_label = self.dataset.features["label"].num_classes > 2 if self.dataset.features["label"] else np.max(self.dataset["label"]) > 1
        self.num_classes = self.dataset.features["label"].num_classes if self.dataset.features["label"] else np.max(self.dataset["label"]) + 1
        self.tokenizer = BertTokenizerForDocumentClassification(model_name, self.multi_label, self.max_length)

    def map(self):
        def tokenization_fn(examples):
            tokenizer = self.tokenizer.tokenizer
            return self.tokenizer.encode_texts(examples["text"], examples["label"])
        
        if self.multi_label:
            dataset = self.dataset.map(
                partial(multi_label_encoding, num_classes=self.num_classes),
                batched=False,
                num_proc=4
            ) # Batching must be false as multi_label_encoding can only process one example at a time
        dataset = dataset.map(tokenization_fn, batched=True, num_proc=4)
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return dataset

"""
dataset = load_dataset("ag_news", split=["train", "test"])[0] # Get train dataset
num_classes = dataset.features["label"].num_classes if dataset.features["label"] else np.max(dataset["label"]) + 1
dataset = dataset.map(partial(multi_label_encoding, num_classes=num_classes))
tokenizer = BertTokenizerForDocumentClassification(multi_label=num_classes > 2, max_length=512, truncation=True)
def tokenize_function(examples):
    return tokenizer.encode_texts(examples["text"], padding="max_length", truncation=True, max_length=512)

dataset = dataset.map(tokenize_function, batched=True)
dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "multi_label"])
print(dataset[0])"
"""

if __name__ == "__main__":
    dataset_processor = BertDatasetForDocumentClassification()
    dataset = dataset_processor.map()
    print(dataset[0])