import torch
from model import LSTMStudent 
from dataset import BertDatasetForDocumentClassification
from transformers import DataCollatorWithPadding

def get_student_predictions(model, dataloader, device):
    model.eval()
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            logits = model(input_ids)
            all_logits.append(logits)

    preds =  torch.cat(all_logits, dim=0) # Combine all logits batchwise
    # Convert predictions to class labels
    return preds

if __name__ == "__main__":
    ltsm_model = LSTMStudent(vocab_size=30522, embedding_dim=768, hidden_dim=256, num_classes=4)
    dataset_processor = BertDatasetForDocumentClassification(split=["test"])
    dataset = dataset_processor.map()
    tokenizer = dataset_processor.tokenizer.tokenizer

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    demo_dataset_part = dataset.select(range(10))
    demo_loader = torch.utils.data.DataLoader(demo_dataset_part, batch_size=2, collate_fn=data_collator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student_predictions = get_student_predictions(ltsm_model, demo_loader, device)
    student_predictions = torch.argmax(student_predictions, dim=1)
    
    for i in range(10):
        print(f"Sentence: {tokenizer.decode(demo_dataset_part['input_ids'][i])}")
        print(f"Prediction: {student_predictions[i]}")
        print(f"True label: {demo_dataset_part['labels'][i]}")
        print("\n")