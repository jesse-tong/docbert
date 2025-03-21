from transformers import BertTokenizer
import torch

class BertTokenizerForDocumentClassification:
    def __init__(self, model_name="bert-base-uncased", multi_label=False, max_length=512, truncation=True):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.multi_label = multi_label
        self.max_length = max_length
        self.truncation = truncation

    def encode_texts(self, texts, labels):
        encodings = self.tokenizer(texts, padding=True, truncation=self.truncation, max_length=self.max_length, return_tensors="pt")
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(labels, dtype=torch.float if self.multi_label else torch.long)
        }




