from dataset import BertDatasetForDocumentClassification
from model import BertForDocumentClassification
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import DataCollatorWithPadding
from tqdm import tqdm, trange
import argparse, evaluate

def evaluate_model(model, eval_loader, device):
    with torch.no_grad():
        model.eval()
        all_logits = []
        all_labels = []
        for batch in tqdm(eval_loader, leave=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            logits = model(input_ids, attention_mask)
            all_logits.append(logits)
            all_labels.append(labels)

        accuracy = evaluate.load("accuracy")
        f1 = evaluate.load("f1")

        # Change in evaluate_model
        if multi_label:
            predictions = torch.argmax(all_logits, dim=1)
            true_labels = torch.argmax(all_labels, dim=1)
        else:
            predictions = torch.argmax(all_logits, dim=1)
            true_labels = all_labels  # Labels are already indices

        acc = accuracy.compute(predictions=predictions, references=true_labels)
        f1_score = f1.compute(predictions=predictions, references=true_labels, average="micro")

        return acc, f1_score

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, leave=True):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--dataset", type=str, default="ag_news")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save_path", type=str, default="bert_document_classification")
    args = parser.parse_args()
    
    dataset_processor = BertDatasetForDocumentClassification(args.dataset, args.model_name)
    dataset = dataset_processor.map()
    tokenizer = dataset_processor.tokenizer.tokenizer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    num_classes = dataset_processor.num_classes
    multi_label = dataset_processor.multi_label

    eval_dataset = BertDatasetForDocumentClassification(args.dataset, args.model_name, split=["test"])
    # Get about 320 samples for evaluation
    eval_dataset.dataset = eval_dataset.dataset.select(range(320) if len(eval_dataset.dataset) > 320 else range(len(eval_dataset.dataset)))
    eval_dataset = eval_dataset.map()
    eval_loader = DataLoader(eval_dataset, batch_size=16, collate_fn=data_collator)
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=data_collator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BertForDocumentClassification(args.model_name, num_classes, multi_label).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss() if not multi_label else nn.BCEWithLogitsLoss()
    
    for epoch in trange(args.epochs, leave=True):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss}")
        acc, f1 = evaluate_model(model, eval_loader, device)
        print(f"Epoch {epoch + 1} - Eval Accuracy: {acc} - F1 Score: {f1}") 

    model.save_pretrained(args.save_path)