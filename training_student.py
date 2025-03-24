from model import LSTMStudent, KnowledgeDistillationLoss
from dataset import BertDatasetForDocumentClassification
from tokenizer import DocumentBatchCollator
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertForSequenceClassification
import argparse
from tqdm import tqdm, trange

def get_teacher_predictions(model, dataloader, device):
    model.eval()
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask).logits
            all_logits.append(logits)

    return torch.cat(all_logits, dim=0) # Combine all logits batchwise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert_document_classification")
    parser.add_argument("--dataset", type=str, default="ag_news")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--student_save_path", type=str, default="lstm_document_classification")
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    dataset_processor = BertDatasetForDocumentClassification(args.dataset, "bert-base-uncased")
    dataset = dataset_processor.map()
    
    tokenizer = dataset_processor.tokenizer.tokenizer
    data_collator = DocumentBatchCollator(pad_token_id=tokenizer.pad_token_id)
    num_classes = dataset_processor.num_classes
    multi_label = dataset_processor.multi_label
    print("Dataset processed!")

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    teacher_model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=num_classes).to(device)
    teacher_model.eval()

    student_model = LSTMStudent(vocab_size=30522, embedding_dim=768, hidden_dim=256, num_classes=num_classes).to(device)
    optimizer = optim.AdamW(student_model.parameters(), lr=args.lr)
    criterion = KnowledgeDistillationLoss(temperature=args.temperature, alpha=args.alpha)

    print("Getting teacher predictions...")
    teacher_predictions = get_teacher_predictions(teacher_model, train_loader, device)
    print("Teacher predictions obtained!")

    for epoch in trange(args.epochs):
        student_model.train()
        total_loss = 0
        for i, batch in enumerate(tqdm(train_loader)):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_masks = batch["attention_mask"].to(device)

            with torch.no_grad():
                teacher_logits = teacher_model.bert(input_ids=input_ids, attention_mask=attention_masks).last_hidden_state

            optimizer.zero_grad()
            student_logits = student_model(input_ids)
            loss = criterion(student_logits, teacher_predictions[i * args.batch_size: (i + 1) * args.batch_size], labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1} - Train Loss: {total_loss / len(train_loader)}")

    # Save LSTM student model
    torch.save(student_model.state_dict(), args.student_save_path)