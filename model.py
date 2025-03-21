import torch
import torch.nn as nn
from transformers import BertModel

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=2.0, alpha=0.5):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, labels):
        # Compute soft targets using temperature
        student_probs = torch.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=-1)
        
        # KL Divergence loss
        distill_loss = self.kl_loss(student_probs, teacher_probs)

        # Standard classification loss
        classification_loss = self.ce_loss(student_logits, labels)

        # Combine losses
        loss = (1 - self.alpha) * classification_loss + self.alpha * distill_loss
        return loss


class BertForDocumentClassification(nn.Module):
    def __init__(self, model_name="bert-base-uncased", num_classes=2, multi_label=False, dropout=0.2):
        super(BertForDocumentClassification, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob if dropout is None else dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes) # Add a fully connected layer, according to the paper
        self.multi_label = multi_label

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_hidden_state = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_hidden_state)
        logits = self.fc(x)

        if self.multi_label:
            logits = torch.sigmoid(logits)
        else:
            logits = torch.softmax(logits, dim=1)

        return logits
    
class LSTMStudent(nn.Module):
    def __init__(self, vocab_size, embedding_dim=768, hidden_dim=256, num_classes=4, 
                 num_layers=2, dropout=0.2):
        super(LSTMStudent, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        final_hidden_state = lstm_out[:, -1, :]
        logits = self.fc(final_hidden_state)
        return torch.sigmoid(logits)

