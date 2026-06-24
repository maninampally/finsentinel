import mlflow
import mlflow.pytorch
import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AdamW,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class FinancialSentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_finbert(use_databricks=False, spark=None):
    if use_databricks and spark:
        # Read from Databricks Delta table
        spark_df = spark.sql("""
            SELECT cleaned_text AS sentence, 1 AS label
            FROM finsentinel.silver.articles_silver
            WHERE length(cleaned_text) >= 20
            LIMIT 2769
        """)
        df = spark_df.toPandas()
    else:
        # Fallback: use HuggingFace dataset
        dataset = load_dataset("financial_phrasebank", "sentences_allagree")
        df = dataset['train'].to_pandas()

    X_train, X_val, y_train, y_val = train_test_split(
        df['sentence'].tolist(), df.get('label', [1]*len(df)).tolist(),
        test_size=0.2, stratify=df.get('label', [1]*len(df)), random_state=42
    )

    tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
    model = BertForSequenceClassification.from_pretrained(
        'ProsusAI/finbert', num_labels=3
    ).to(DEVICE)

    train_loader = DataLoader(
        FinancialSentimentDataset(X_train, y_train, tokenizer),
        batch_size=16, shuffle=True
    )
    val_loader = DataLoader(
        FinancialSentimentDataset(X_val, y_val, tokenizer),
        batch_size=32
    )

    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) // 4,
        num_training_steps=len(train_loader) * 5
    )

    mlflow.set_experiment("finsentinel-sentiment")

    with mlflow.start_run(run_name="finbert_finetuned"):
        mlflow.log_params({
            "model":        "ProsusAI/finbert",
            "epochs":       5,
            "lr":           2e-5,
            "batch_size":   16,
            "max_len":      128,
            "dataset":      "FinancialPhraseBank",
            "dataset_size": len(df)
        })

        for epoch in range(5):
            model.train()
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch['input_ids'].to(DEVICE),
                    attention_mask=batch['attention_mask'].to(DEVICE),
                    labels=batch['label'].to(DEVICE)
                )
                outputs.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += outputs.loss.item()

            model.eval()
            preds, actuals = [], []
            with torch.no_grad():
                for batch in val_loader:
                    out = model(
                        input_ids=batch['input_ids'].to(DEVICE),
                        attention_mask=batch['attention_mask'].to(DEVICE)
                    )
                    preds.extend(out.logits.argmax(-1).cpu().numpy())
                    actuals.extend(batch['label'].numpy())

            f1 = f1_score(actuals, preds, average='weighted')
            avg_loss = total_loss / len(train_loader)
            mlflow.log_metrics({"train_loss": avg_loss, "val_f1": f1}, step=epoch)
            print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | F1: {f1:.4f}")

        print(classification_report(
            actuals, preds,
            target_names=['Negative', 'Neutral', 'Positive']
        ))

        mlflow.pytorch.log_model(
            model, "finbert_model",
            registered_model_name="FinSentinel_Production"
        )
        mlflow.log_metric("final_f1", f1)

    return model, tokenizer


if __name__ == '__main__':
    train_finbert()
