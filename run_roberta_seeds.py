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

BASE_DIR = os.environ.get("HOME", os.getcwd())
DATA_PATH = os.path.join(BASE_DIR, "DeepLearning/project/deep_learning_project/data", "data_huang_devansh_processed_equal.csv")
FINAL_RESULTS_DIR = os.path.join(BASE_DIR, "DeepLearning/project/deep_learning_project/results/results_roberta_seeds")
MODEL_DIR = os.path.join(BASE_DIR, "DeepLearning/project/deep_learning_project/models/models_roberta_seeds")
os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 101112]
all_metrics = []

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

for seed in SEEDS:
    print(f"\n===Running for SEED {seed} ===")

    RESULTS_DIR = os.path.join(FINAL_RESULTS_DIR, f"results_seed{seed}")
    MODEL_DIR = os.path.join(MODEL_DIR, f"model_seed{seed}")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

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

    df = pd.DataFrame(raw_rows, columns=["text", "label"]).dropna()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df = df.head(100000)

    split_idx = int(len(df) * 0.7)
    train_df, eval_df = df[:split_idx], df[split_idx:]

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")
    train_encodings = tokenizer(train_df["text"].tolist(), truncation=True, padding=True, max_length=128)
    eval_encodings = tokenizer(eval_df["text"].tolist(), truncation=True, padding=True, max_length=128)

    train_dataset = HateSpeechDataset(train_encodings, train_df["label"].tolist())
    eval_dataset = HateSpeechDataset(eval_encodings, eval_df["label"].tolist())

    model = AutoModelForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-offensive",
        num_labels=2,
        device_map="auto",
        use_safetensors=True
    )

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

    training_args = TrainingArguments(
        output_dir=RESULTS_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=100,
        weight_decay=0.01,
        learning_rate=2e-5,
        logging_dir=os.path.join(BASE_DIR, "logs"),
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        report_to="none",
        seed=seed,
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

    trainer.train()
    metrics = trainer.evaluate()
    all_metrics.append({"seed": seed, **metrics})

    pd.DataFrame([
        {"metric": "accuracy", "value": metrics["eval_accuracy"]},
        {"metric": "f1_Hate", "value": metrics["eval_f1_Hate"]},
        {"metric": "f1_NoHate", "value": metrics["eval_f1_NoHate"]}
    ]).to_csv(os.path.join(RESULTS_DIR, "roberta_eval_metrics.csv"), index=False)

    pd.DataFrame({
        "step": train_metrics["step"],
        "train_acc": train_metrics["accuracy"],
        "train_loss": train_metrics["loss"]
    }).to_csv(os.path.join(RESULTS_DIR, "roberta_train_progress.csv"), index=False)

    plt.figure()
    plt.bar(["Accuracy", "F1_Hate", "F1_NoHate"], [
        metrics["eval_accuracy"],
        metrics["eval_f1_Hate"],
        metrics["eval_f1_NoHate"]
    ])
    plt.ylim(0, 1)
    plt.title(f"Eval Metrics Seed {seed}")
    plt.savefig(os.path.join(RESULTS_DIR, "barplot_metrics.png"))
    plt.close()

    plt.figure()
    plt.plot(train_metrics["step"], train_metrics["loss"], label="Train Loss")
    plt.plot(train_metrics["step"], train_metrics["accuracy"], label="Train Accuracy")
    plt.xlabel("Step")
    plt.legend()
    plt.title(f"Training Progress Seed {seed}")
    plt.savefig(os.path.join(RESULTS_DIR, "training_progress.png"))
    plt.close()

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

# ===========================
# Aggregate Results
# ===========================

df_all = pd.DataFrame(all_metrics)
agg = df_all[["eval_accuracy", "eval_f1_Hate", "eval_f1_NoHate"]].agg(["mean", "std", "var"])
agg.to_csv(os.path.join(FINAL_RESULTS_DIR, "final_summary.csv"))

print("All seeds completed. Aggregated results saved to `results_all/final_summary.csv`.")
