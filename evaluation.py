from model import LSTMStudent, BertForDocumentClassification
from dataset import BertDatasetForDocumentClassification
from transformers import DataCollatorWithPadding
import argparse, torch, evaluate
from inference import get_student_predictions
from training_student import get_teacher_predictions

# Convert predictions to class labels
def convert_preds_to_labels(preds):
    return torch.argmax(preds, dim=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ltsm_save_path", type=str, default="lstm_document_classification")
    parser.add_argument("--bert_save_path", type=str, default="bert_document_classification")
    parser.add_argument("--dataset", type=str, default="ag_news")
    parser.add_argument("--eval_size", type=int, default=100)
    args = parser.parse_args()

    dataset_processor = BertDatasetForDocumentClassification(dataset=args.dataset, split=["test"])
    dataset = dataset_processor.map()
    dataset_size = len(dataset)
    tokenizer = dataset_processor.tokenizer.tokenizer

    num_classes = dataset_processor.num_classes
    multi_label = dataset_processor.multi_label

    ltsm_model = LSTMStudent(vocab_size=30522, embedding_dim=768, hidden_dim=256, num_classes=num_classes)
    ltsm_model.load_state_dict(torch.load(args.ltsm_save_path))

    bert_model = BertForDocumentClassification(args.bert_save_path, num_classes=num_classes, multi_label=multi_label)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    eval_dataset_part = dataset.select(range(args.eval_size) if int(args.eval_size) < dataset_size else range(dataset_size))
    batch_size = 16 if int(args.eval_size) > 16 else int(args.eval_size)

    eval_loader = torch.utils.data.DataLoader(eval_dataset_part, batch_size=batch_size, collate_fn=data_collator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ltsm_predictions = get_student_predictions(ltsm_model, eval_loader, device)
    ltsm_predictions = convert_preds_to_labels(ltsm_predictions)

    bert_predictions = get_teacher_predictions(bert_model, eval_loader, device)
    bert_predictions = convert_preds_to_labels(bert_predictions)


    true_labels = eval_dataset_part["labels"]
    true_labels = convert_preds_to_labels(true_labels)

    accuracy = evaluate.load("accuracy")
    f1 = evaluate.load("f1")

    ltsm_accuracy = accuracy.compute(predictions=ltsm_predictions, references=true_labels)
    bert_accuracy = accuracy.compute(predictions=bert_predictions, references=true_labels)

    ltsm_f1 = f1.compute(predictions=ltsm_predictions, references=true_labels, average="micro")
    bert_f1 = f1.compute(predictions=bert_predictions, references=true_labels, average="micro")

    print(f"LSTM Student Accuracy: {ltsm_accuracy}")
    print(f"Bert Model Accuracy: {bert_accuracy}")

    print(f"LSTM Student F1 Score: {ltsm_f1}")
    print(f"Bert Model F1 Score: {bert_f1}")
    
    

