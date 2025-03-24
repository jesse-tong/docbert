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
    
class LSTMStudent(nn.Module):
    def __init__(self, vocab_size, embedding_dim=300, hidden_dim=512, num_classes=4, 
                  dropout=0.2):
        super(LSTMStudent, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Regularization
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True, num_layers=1)
        self.fc = nn.Linear(hidden_dim * 2, num_classes) # BiLSTM has 2x hidden_dim

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        
        # BiLSTM processing
        lstm_out, (hidden, _) = self.lstm(embedded)
        
        # shape: [batch_size, hidden_dim*2]
        final_hidden = torch.cat((hidden[0], hidden[1]), dim=1)
        final_hidden = self.dropout(final_hidden)
        logits = self.fc(final_hidden)

        return logits

