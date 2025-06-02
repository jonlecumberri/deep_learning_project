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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
BASE_DIR = os.environ.get("HOME", os.getcwd())
DATA_PATH = os.path.join(BASE_DIR, "DeepLearning/project/data", "data_huang_devansh_processed_equal.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "DeepLearning/project/results", "results_qwen_chat_fewshot_trimmed")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ===========================
# Suppress Warnings
# ===========================

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
logging.set_verbosity_error()

# ===========================
# Load Data
# ===========================

print(f"📄 Loading data from: {DATA_PATH}")

raw_rows = []
with open(DATA_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, quoting=csv.QUOTE_NONE)
    for row in reader:
        text = (row.get("Content") or "").strip()
        label = (row.get("normalized_label") or "").strip()
        if text and label in {"0", "1"}:
            raw_rows.append((text, int(label)))

df = pd.DataFrame(raw_rows, columns=["text", "label"])
print(f"🥪 Size of FULL valid dataset: {len(df)}")
print(f"🔍 Number of NoHate: {(df['label'] == 0).sum()} | Hate: {(df['label'] == 1).sum()}")

df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.head(100000)

# ===========================
# Few-shot Prompt from Dataset (3 Hate + 3 NoHate)
# ===========================

instruction = (
    "You are a content moderation assistant. Classify each text as Hate '1' or No Hate '0'. "
    "Reply only with the number.\n\n"
)

example_id = 1
hate_examples = df[df["label"] == 1].head(3).reset_index(drop=True)
nohate_examples = df[df["label"] == 0].head(3).reset_index(drop=True)
shot_rows = pd.concat([hate_examples, nohate_examples]).sample(frac=1, random_state=123).reset_index(drop=True)

example_prompt = ""
for row in shot_rows.itertuples(index=False):
    label_text = "1" if row.label == 1 else "0"
    example_prompt += f"Example {example_id}:\n{row.text}\n{label_text}\n"
    example_id += 1

instruction += example_prompt + "Now classify:\n"
instruction = "hello"

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

def run_prompt_inference(df, model, tokenizer):
    results = []
    print(f"\n🚀 Running few-shot inference on {len(df)} samples...")

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
        text = row["text"]
        gold_label = int(row["label"])
        gold_str = "Hate" if gold_label == 1 else "NoHate"

        prompt = instruction + f"{text}\nAnswer:"
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        if input_ids.shape[1] > 2000:
            prompt = (
                "You are a content moderation assistant. Classify the following text as Hate (1) or No Hate (0). "
                "Reply with 0 or 1 only.\n\n"
                f"Text: {text}\nAnswer:"
            )
            prompt = "Hello"
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
            )
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        print(decoded)
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

        if i < 5:
            print("\n🔍 DEBUG SAMPLE")
            print("Prompt:\n", prompt)
            print("Raw Output:\n", decoded)
            print(f"Parsed Answer: {answer}")
            print(f"Predicted: {pred_label} | Ground Truth: {gold_str}")

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
# Run Inference
# ===========================

eval_df = df.sample(n=1, random_state=123)
few_shot_df = run_prompt_inference(eval_df, model, tokenizer)
few_shot_df.to_csv(os.path.join(RESULTS_DIR, "few_shot_predictions.csv"), index=False)

# ===========================
# Save Metrics
# ===========================

def save_metrics(df):
    acc = accuracy_score(df["gold_label"], df["pred_num"])
    report = classification_report(
        df["gold_label"], df["pred_num"], target_names=["NoHate", "Hate"], output_dict=True, zero_division=0
    )
    metrics = pd.DataFrame([{
        "accuracy": acc,
        "f1_Hate": report["Hate"]["f1-score"],
        "f1_NoHate": report["NoHate"]["f1-score"]
    }])
    metrics.to_csv(os.path.join(RESULTS_DIR, "few_shot_metrics.csv"), index=False)
    print(f"\n📊 FEW_SHOT METRICS — Accuracy: {acc:.4f}, F1_Hate: {report['Hate']['f1-score']:.4f}, F1_NoHate: {report['NoHate']['f1-score']:.4f}")
    print(f"📊 PREDICTION DISTRIBUTION —\n{df['pred_label'].value_counts()}\n")

save_metrics(few_shot_df)
print("\n🚀 Few-shot inference complete.")
