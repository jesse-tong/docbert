from transformers import BertTokenizer
import torch
from torch.nn.utils.rnn import pad_sequence

class DocumentBatchCollator:
    def __init__(self, pad_token_id=0):
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch):
        """
        Input batch is a list of tuples: (input_ids, label)
        Returns a batch ready for BertForSequenceClassification
        """
        input_ids_list = [item['input_ids'] for item in batch]
        labels = torch.stack([item['labels'] for item in batch])
        
        # Pad sequences
        input_ids_padded = pad_sequence(
            input_ids_list, 
            batch_first=True, 
            padding_value=self.pad_token_id
        )
        
        # Create attention masks (1 for real tokens, 0 for padding)
        attention_mask = (input_ids_padded != self.pad_token_id).float()
        
        # If you want to customize your attention masks beyond padding, do it here
        # Example: custom_attention = modify_attention_mask(attention_mask, input_ids_padded)
        
        return {
            'input_ids': input_ids_padded,
            'attention_mask': attention_mask,
            'labels': labels
        }

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




