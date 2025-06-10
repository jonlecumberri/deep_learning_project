import os
import csv
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, classification_report
from transformers import logging
import warnings
from sklearn.exceptions import UndefinedMetricWarning

# ===========================
# Setup
# ===========================

BASE_DIR = os.environ.get("HOME", os.getcwd())
DATA_PATH = os.path.join(BASE_DIR, "DeepLearning/project/deep_learning_project/data", "data_huang_devansh_processed_equal.csv")
FINAL_RESULTS_DIR = os.path.join(BASE_DIR, "DeepLearning/project/deep_learning_project/results/results_qwen_chat_fewshot_seeds")
os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)

SEEDS = [42, 123, 456, 789, 101112]
all_metrics = []

# ===========================
# Suppress Warnings
# ===========================

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
logging.set_verbosity_error()

# ===========================
# Load Model
# ===========================

model_name = "Qwen/Qwen1.5-0.5B-Chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cuda()
model.eval()

# ===========================
# Inference Function
# ===========================

def run_prompt_inference(df, model, tokenizer, instruction):
    results = []
    print(f"\nRunning few-shot inference on {len(df)} samples...")

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        text = row["text"]
        gold_label = int(row["label"])
        gold_str = "Hate" if gold_label == 1 else "NoHate"

        prompt = instruction + f"{text}\nAnswer:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        if input_ids.shape[1] > 2000:
            prompt = (
                "You are a content moderation assistant. Classify the following text as Hate (1) or No Hate (0). "
                "Reply with 0 or 1 only.\n\nText: " + text + "\nAnswer:"
            )
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids)
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        answer = decoded.split("Answer:")[-1].strip() if "Answer:" in decoded else decoded.strip().split()[-1].strip()
        answer = answer.replace("'", "").replace('"', '').lower()

        if answer.startswith("1"):
            pred_label = "Hate"
            pred_num = 1
        elif answer.startswith("0"):
            pred_label = "NoHate"
            pred_num = 0
        else:
            pred_label = "NoHate"
            pred_num = 0

        results.append({
            "text": text,
            "gold_label": gold_label,
            "gold_str": gold_str,
            "pred_num": pred_num,
            "pred_label": pred_label,
            "raw_output": decoded,
            "parsed_answer": answer
        })

    return pd.DataFrame(results)

# ===========================
# Loop Over Seeds
# ===========================

for seed in SEEDS:
    print(f"\n===Few-shot for SEED {seed} ===")
    seed_dir = os.path.join(FINAL_RESULTS_DIR, f"results_seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    # Load and sample data
    raw_rows = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, quoting=csv.QUOTE_NONE)
        for row in reader:
            text = (row.get("Content") or "").strip()
            label = (row.get("normalized_label") or "").strip()
            if text and label in {"0", "1"}:
                raw_rows.append((text, int(label)))
    df = pd.DataFrame(raw_rows, columns=["text", "label"])
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True).head(100000)

    # Build few-shot prompt
    hate_examples = df[df["label"] == 1].head(3).reset_index(drop=True)
    nohate_examples = df[df["label"] == 0].head(3).reset_index(drop=True)
    shot_rows = pd.concat([hate_examples, nohate_examples]).sample(frac=1, random_state=123).reset_index(drop=True)

    instruction = (
        "You are a content moderation assistant. Classify each text as Hate '1' or No Hate '0'. Reply only with the number.\n\n"
    )
    for idx, row in enumerate(shot_rows.itertuples(index=False), start=1):
        label_text = "1" if row.label == 1 else "0"
        instruction += f"Example {idx}:\n{row.text}\n{label_text}\n"
    instruction += "Now classify:\n"

    # Sample data for evaluation
    eval_df = df.sample(n=100, random_state=seed)
    few_shot_df = run_prompt_inference(eval_df, model, tokenizer, instruction)
    few_shot_df.to_csv(os.path.join(seed_dir, "few_shot_predictions.csv"), index=False)

    acc = accuracy_score(few_shot_df["gold_label"], few_shot_df["pred_num"])
    report = classification_report(few_shot_df["gold_label"], few_shot_df["pred_num"], target_names=["NoHate", "Hate"], output_dict=True, zero_division=0)
    all_metrics.append({
        "seed": seed,
        "eval_accuracy": acc,
        "eval_f1_Hate": report["Hate"]["f1-score"],
        "eval_f1_NoHate": report["NoHate"]["f1-score"]
    })

    metrics_df = pd.DataFrame([{ "accuracy": acc, "f1_Hate": report["Hate"]["f1-score"], "f1_NoHate": report["NoHate"]["f1-score"] }])
    metrics_df.to_csv(os.path.join(seed_dir, "few_shot_metrics.csv"), index=False)

# ===========================
# Aggregate Results
# ===========================

df_all = pd.DataFrame(all_metrics)
agg = df_all[["eval_accuracy", "eval_f1_Hate", "eval_f1_NoHate"]].agg(["mean", "std", "var"])
agg.to_csv(os.path.join(FINAL_RESULTS_DIR, "final_summary.csv"))

print("\nAll few-shot seeds completed. Aggregated results saved to `results_qwen_chat_fewshot_seeds/final_summary.csv`.")
