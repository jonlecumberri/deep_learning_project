import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup paths based on the script location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("Base directory:", BASE_DIR)
RESULTS_DIR = os.path.join(BASE_DIR, "results", "eda")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Loading dataset...")
df2 = pd.read_csv(os.path.join(DATA_DIR, "data_huang_devansh.csv"))
df2['normalized_label'] = df2['Label']  # already binary

# Save basic info
print("Saving dataset info and label distribution...")
with open(os.path.join(RESULTS_DIR, "dataset_info.txt"), "w") as f:
    f.write("\n\n--- Dataset 2 ---\n")
    df2.info(buf=f)
    f.write("\nLabel distribution:\n")
    f.write(df2['normalized_label'].value_counts().to_string())

# Barplot: Label distribution
print("Plotting label distribution (original)...")
label_counts = df2['normalized_label'].value_counts().sort_index()
label_counts.to_csv(os.path.join(RESULTS_DIR, "label_dist_df2.csv"))
sns.countplot(x='normalized_label', data=df2)
plt.title("Label Distribution in data_huang_devansh.csv")
plt.savefig(os.path.join(RESULTS_DIR, "label_dist_df2.png"))
plt.clf()

# Histogram: Text length
print("Calculating text lengths and plotting histogram (original)...")
df2['text_length'] = df2['Content'].astype(str).apply(len)
text_length_counts = df2['text_length'].value_counts().sort_index()
text_length_counts.to_csv(os.path.join(RESULTS_DIR, "content_length_df2.csv"))

sns.histplot(df2['text_length'], bins=30, kde=True)
plt.title("Content Length Distribution - Dataset 2")
plt.xlabel("Length")
plt.savefig(os.path.join(RESULTS_DIR, "content_length_df2.png"))
plt.clf()

# Save descriptive stats
df2_stats = df2['text_length'].describe()
with open(os.path.join(RESULTS_DIR, "text_length_stats.txt"), "w") as f:
    f.write("\n\n--- Dataset 2 Content Length Stats ---\n")
    f.write(df2_stats.to_string())

# Filter: Remove long entries
print("Filtering dataset for length < 500...")
df2_filtered = df2[df2['text_length'] < 500]
filtered_path = os.path.join(DATA_DIR, "data_huang_devansh_processed2.csv")
df2_filtered.to_csv(filtered_path, index=False)
print(f"Filtered dataset saved to: {filtered_path}")

# Label dist (filtered)
print("Plotting label distribution (filtered)...")
filtered_label_counts = df2_filtered['normalized_label'].value_counts().sort_index()
filtered_label_counts.to_csv(os.path.join(RESULTS_DIR, "label_dist_df2_filtered.csv"))
sns.countplot(x='normalized_label', data=df2_filtered)
plt.title("Label Distribution in data_huang_devansh_filtered.csv")
plt.savefig(os.path.join(RESULTS_DIR, "label_dist_df2_filtered.png"))
plt.clf()

# Text length dist (filtered)
print("Plotting content length histogram (filtered)...")
filtered_text_length_counts = df2_filtered['text_length'].value_counts().sort_index()
filtered_text_length_counts.to_csv(os.path.join(RESULTS_DIR, "content_length_df2_filtered.csv"))

sns.histplot(df2_filtered['text_length'], bins=30, kde=True)
plt.title("Content Length Distribution - Dataset 2 Filtered")
plt.xlabel("Length")
plt.savefig(os.path.join(RESULTS_DIR, "content_length_df2_filtered.png"))
plt.clf()

df2_filtered_stats = df2_filtered['text_length'].describe()
with open(os.path.join(RESULTS_DIR, "text_length_stats.txt"), "a") as f:
    f.write("\n\n--- Dataset 2 Content Length Stats (Filtered) ---\n")
    f.write(df2_filtered_stats.to_string())

# Balance dataset
print("Balancing filtered dataset (50-50)...")
label_0 = df2_filtered[df2_filtered['normalized_label'] == 0]
label_1 = df2_filtered[df2_filtered['normalized_label'] == 1]
min_count = min(len(label_0), len(label_1))
balanced_df = pd.concat([
    label_0.sample(n=min_count, random_state=42),
    label_1.sample(n=min_count, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

balanced_path = os.path.join(DATA_DIR, "data_huang_devansh_processed_equal2.csv")
balanced_df.to_csv(balanced_path, index=False)
print(f"Balanced dataset (50-50) saved to: {balanced_path}")

# Plot and save label dist (balanced)
print("Plotting label distribution (filtered + balanced)...")
balanced_label_counts = balanced_df['normalized_label'].value_counts().sort_index()
balanced_label_counts.to_csv(os.path.join(RESULTS_DIR, "label_dist_df2_filtered_equal.csv"))

sns.countplot(x='normalized_label', data=balanced_df)
plt.title("Label Distribution in data_huang_devansh_filtered_equal.csv")
plt.savefig(os.path.join(RESULTS_DIR, "label_dist_df2_filtered_equal.png"))
plt.clf()

# Plot and save text length (balanced)
print("Plotting content length histogram (filtered + balanced)...")
balanced_text_length_counts = balanced_df['text_length'].value_counts().sort_index()
balanced_text_length_counts.to_csv(os.path.join(RESULTS_DIR, "content_length_df2_filtered_equal.csv"))

sns.histplot(balanced_df['text_length'], bins=30, kde=True)
plt.title("Content Length Distribution - Dataset 2 Filtered Balanced")
plt.xlabel("Length")
plt.savefig(os.path.join(RESULTS_DIR, "content_length_df2_filtered_equal.png"))
plt.clf()

# Save stats
balanced_stats = balanced_df['text_length'].describe()
with open(os.path.join(RESULTS_DIR, "text_length_stats.txt"), "a") as f:
    f.write("\n\n--- Dataset 2 Content Length Stats (Filtered & Balanced 50-50) ---\n")
    f.write(balanced_stats.to_string())
    f.write("\nLabel distribution (should be 50-50):\n")
    f.write(balanced_label_counts.to_string())

print("EDA complete. All results saved in the 'results/eda' directory.")
