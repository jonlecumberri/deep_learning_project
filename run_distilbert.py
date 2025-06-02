import os
import sys
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback
)

# ===========================
# Setup and Logging
# ===========================

BASE_DIR = os.environ.get("HOME", os.getcwd())
RESULTS_DIR = os.path.join(BASE_DIR, "DeepLearning/project/results", "results_bert")
MODEL_DIR = os.path.join(BASE_DIR, "DeepLearning/project/models", "distilbert_model")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

log_path = os.path.join(RESULTS_DIR, "training_log.txt")
log_file = open(log_path, "w", encoding="utf-8")

class Logger(object):
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = sys.stderr = Logger(sys.stdout, log_file)

print("📁 Base directory:", BASE_DIR)

# ===========================
# Data Preparation
# ===========================

DATA_PATH = os.path.join(BASE_DIR, "DeepLearning/project/data", "data_huang_devansh_processed_equal.csv")
print(f"📄 Loading data from: {DATA_PATH}")

raw_rows = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, quoting=csv.QUOTE_NONE)
    for row in reader:
        if row is None:
            continue  # Skip completely malformed rows
        text = (row.get("Content") or "").strip()
        label = (row.get("normalized_label") or "").strip()
        if text and label in {"0", "1"}:
            raw_rows.append((text, int(label)))

df = pd.DataFrame(raw_rows, columns=["text", "label"])
print(f"🧪 Size of FULL valid dataset: {len(df)}")
print(f"🔍 Number of NoHate: {(df['label'] == 0).sum()} | Hate: {(df['label'] == 1).sum()}")

# df = pd.read_csv(
#     DATA_PATH,
#     on_bad_lines='skip',
#     quoting=csv.QUOTE_NONE,
#     encoding="utf-8",
#     engine="python"
# )[["Content", "normalized_label"]].rename(columns={"Content": "text", "normalized_label": "label"})

df = df.dropna(subset=["text", "label"])
df["text"] = df["text"].astype(str)
df = df[df["label"].astype(str).str.strip().isin(["0", "1"])].copy()
df["label"] = df["label"].astype(int)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
print(f"🧪 Size of FULL filtered dataset {len(df)}")
df = df.head(100000)
print(f"🔍 Number of NoHate: {(df['label'] == 0).sum()} | Hate: {(df['label'] == 1).sum()} after filter")

split_idx = int(len(df) * 0.8)
train_df, eval_df = df[:split_idx], df[split_idx:]

print(f"✅ Loaded {len(df)} samples.")
print(f"🧪 Train size: {len(train_df)} | Eval size: {len(eval_df)}")

# ===========================
# Tokenization
# ===========================

print("🔤 Tokenizing data...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
train_encodings = tokenizer(train_df["text"].tolist(), truncation=True, padding=True, max_length=128)
eval_encodings = tokenizer(eval_df["text"].tolist(), truncation=True, padding=True, max_length=128)

# ===========================
# Dataset Setup
# ===========================

class HateSpeechDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_dataset = HateSpeechDataset(train_encodings, train_df["label"].tolist())
eval_dataset = HateSpeechDataset(eval_encodings, eval_df["label"].tolist())

# ===========================
# Model and Metrics
# ===========================

print("🧠 Loading DistilBERT model...")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    report = classification_report(labels, preds, target_names=["NoHate", "Hate"], output_dict=True)
    return {
        "accuracy": acc,
        "f1_Hate": report["Hate"]["f1-score"],
        "f1_NoHate": report["NoHate"]["f1-score"]
    }

train_metrics = {"step": [], "accuracy": [], "loss": []}

class TrainEvalCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            metrics = trainer.evaluate(train_dataset)
            print(f"📊 Step {state.global_step} | Train Acc: {metrics['eval_accuracy']:.4f} | Train Loss: {metrics['eval_loss']:.4f}")
            train_metrics["step"].append(state.global_step)
            train_metrics["accuracy"].append(metrics["eval_accuracy"])
            train_metrics["loss"].append(metrics["eval_loss"])

# ===========================
# Training Arguments
# ===========================

training_args = TrainingArguments(
    output_dir=RESULTS_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    gradient_accumulation_steps=1,
    warmup_steps=100,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_dir=os.path.join(BASE_DIR, "logs"),
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    report_to="none",
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[TrainEvalCallback()]
)

# ===========================
# Training & Evaluation
# ===========================

print("🚀 Starting training...")
trainer.train()
print("✅ Training complete. Evaluating...")
metrics = trainer.evaluate()
print("📈 Final metrics:", metrics)

# ===========================
# Save Final Eval Metrics CSV (barplot)
# ===========================

bar_metrics_path = os.path.join(RESULTS_DIR, "distilbert_finetuned_barplot_data.csv")
pd.DataFrame([
    {"metric": "accuracy", "value": metrics["eval_accuracy"]},
    {"metric": "f1_Hate", "value": metrics["eval_f1_Hate"]},
    {"metric": "f1_NoHate", "value": metrics["eval_f1_NoHate"]}
]).to_csv(bar_metrics_path, index=False)

# ===========================
# Plot Final Eval Bar Plot
# ===========================

print("📊 Plotting evaluation metrics barplot...")
plt.figure()
plt.bar(["Accuracy", "F1_Hate", "F1_NoHate"], [
    metrics["eval_accuracy"],
    metrics["eval_f1_Hate"],
    metrics["eval_f1_NoHate"]
])
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("DistilBERT Final Evaluation Metrics")
plt.savefig(os.path.join(RESULTS_DIR, "distilbert_finetuned_barplot.png"))
plt.close()

# ===========================
# Save Training Checkpoints CSV
# ===========================

progress_path = os.path.join(RESULTS_DIR, "distilbert_train_progress_data.csv")
pd.DataFrame({
    "step": train_metrics["step"],
    "train_acc": train_metrics["accuracy"],
    "train_loss": train_metrics["loss"]
}).to_csv(progress_path, index=False)

# ===========================
# Plot Training Checkpoints
# ===========================

print("📊 Plotting loss and accuracy curves...")
plt.figure()
plt.plot(train_metrics["step"], train_metrics["loss"], label="Train Loss")
plt.plot(train_metrics["step"], train_metrics["accuracy"], label="Train Accuracy")
plt.xlabel("Step")
plt.title("DistilBERT Training Checkpoints")
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, "train_progress.png"))
plt.close()

# ===========================
# Save Model and Tokenizer
# ===========================

print("💾 Saving model and tokenizer...")
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print("💾 Saving full metrics...")
metrics_path = os.path.join(RESULTS_DIR, "final_metrics.csv")
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

print("🏁 All done. Logs saved to:", log_path)
log_file.close()
