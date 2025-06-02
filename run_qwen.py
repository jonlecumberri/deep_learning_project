import os
import sys
import pandas as pd
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback
)

# ===========================
# Setup and Logging
# ===========================

BASE_DIR = os.environ.get("HOME", os.getcwd())
RESULTS_DIR = os.path.join(BASE_DIR, "DeepLearning/project/results", "results_qwen")
MODEL_DIR = os.path.join(BASE_DIR, "DeepLearning/project/models", "qwen_model")
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
            continue
        text = (row.get("Content") or "").strip()
        label = (row.get("normalized_label") or "").strip()
        if text and label in {"0", "1"}:
            raw_rows.append((text, int(label)))

df = pd.DataFrame(raw_rows, columns=["text", "label"])
print(f"🥪 Size of FULL valid dataset: {len(df)}")
print(f"🔍 Number of NoHate: {(df['label'] == 0).sum()} | Hate: {(df['label'] == 1).sum()}")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.head(100000)
print(f"🔍 Number of NoHate: {(df['label'] == 0).sum()} | Hate: {(df['label'] == 1).sum()} after filter")

split_idx = int(len(df) * 0.8)
train_df, eval_df = df[:split_idx], df[split_idx:]

print(f"✅ Train size: {len(train_df)} | Eval size: {len(eval_df)}")

# ===========================
# Tokenizer and Model Setup
# ===========================

print("🧠 Loading Qwen model...")
model_name = "Qwen/Qwen1.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# ===========================
# Dataset Class
# ===========================

class HateSpeechDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_encodings = tokenizer(train_df["text"].tolist(), truncation=True, padding=True, max_length=128)
eval_encodings = tokenizer(eval_df["text"].tolist(), truncation=True, padding=True, max_length=128)

train_dataset = HateSpeechDataset(train_encodings, train_df["label"].tolist())
eval_dataset = HateSpeechDataset(eval_encodings, eval_df["label"].tolist())

# ===========================
# Metrics
# ===========================

all_train_loss = []
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            all_train_loss.append(logs["loss"])

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

# ===========================
# TrainingArguments
# ===========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

training_args = TrainingArguments(
    output_dir=RESULTS_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    warmup_steps=100,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_dir=os.path.join(BASE_DIR, "logs"),
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# ===========================
# Trainer
# ===========================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# ===========================
# Train & Evaluate
# ===========================

print("🚀 Starting Qwen fine-tuning...")
trainer.train()
print("✅ Training complete. Evaluating...")
metrics = trainer.evaluate()
print("📈 Final metrics:", metrics)

print("🗞 Barplot values:")
print(f"  Accuracy: {metrics['eval_accuracy']:.4f}")
print(f"  F1_Hate: {metrics['eval_f1_Hate']:.4f}")
print(f"  F1_NoHate: {metrics['eval_f1_NoHate']:.4f}")

bar_metrics_path = os.path.join(RESULTS_DIR, "qwen_finetuned_barplot_data.csv")
pd.DataFrame([
    {"metric": "accuracy", "value": metrics["eval_accuracy"]},
    {"metric": "f1_Hate", "value": metrics["eval_f1_Hate"]},
    {"metric": "f1_NoHate", "value": metrics["eval_f1_NoHate"]}
]).to_csv(bar_metrics_path, index=False)

# ===========================
# Save Results
# ===========================

print("📂 Saving model and tokenizer...")
model.save_pretrained(MODEL_DIR)
tokenizer.save_pretrained(MODEL_DIR)

print("📂 Saving metrics...")
metrics_path = os.path.join(RESULTS_DIR, "qwen_finetuned_metrics.csv")
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

print("📊 Plotting results...")
plt.figure()
plt.bar(["Accuracy", "F1_Hate", "F1_NoHate"], [
    metrics["eval_accuracy"],
    metrics["eval_f1_Hate"],
    metrics["eval_f1_NoHate"]
])
plt.ylim(0, 1)
plt.ylabel("Score")
plt.title("Qwen Fine-Tuned Metrics")
plt.savefig(os.path.join(RESULTS_DIR, "qwen_finetuned_barplot.png"))
plt.close()

# ===========================
# Save and Plot Training Loss
# ===========================

loss_path = os.path.join(RESULTS_DIR, "qwen_finetuned_loss_data.csv")
pd.DataFrame({"step": list(range(len(all_train_loss))), "loss": all_train_loss}).to_csv(loss_path, index=False)

plt.figure()
plt.plot(all_train_loss)
plt.title("Qwen Training Loss")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.savefig(os.path.join(RESULTS_DIR, "qwen_finetuned_loss_plot.png"))
plt.close()

print("🌝 All done. Logs saved to:", log_path)
log_file.close()
