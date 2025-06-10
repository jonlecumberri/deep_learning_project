# From BERT to Qwen: Hate Detection across Architectures

This repository contains code and data for the **EE‑559 Deep Learning** group project at *EPFL*. The objective is to compare several language‑model architectures (BERT, DistilBERT, RoBERTa, Gemma, Qwen) on the task of **hate‑speech detection**.

Authors · *Ariadna Mon Gomis · Saúl Fenollosa Arguedas · Jon Lecumberri Arteta*

---

## Repository Layout

| Path/File                                                                                                            | Purpose                                                                                                                         |
| -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **`data/`**                                                                                                          | Link to the raw dataset (hosted on Google Drive due to size constraints).                                                       |
| **`.gitignore`**                                                                                                     | Excludes generated artefacts (`data/`, `models/`, `results/`) from version control.                                             |
| **`requirements.txt`**                                                                                               | **Only** the dependencies needed to run the *Python scripts* in this repo (everything ending in `.py`).                         |
| **`data_exploration.py`**                                                                                            | Creates the processed & balanced dataset used by all downstream experiments.                                                    |
| **`run_distilbert_seeds.py`**<br>**`run_qwen_seeds.py`**<br>**`run_qwen_ft_seeds.py`**<br>**`run_roberta_seeds.py`** | Command‑line entry points for the core experiments. Each script can be executed with multiple random seeds for reproducibility. |
| **`run_gemma.ipynb`**                                                                                                | *Standalone* Colab notebook for Gemma experiments (see below).                                                                  |

---

## Getting Started (Python Scripts)

> The steps below are **only** required if you plan to execute the `.py` experiment scripts locally. The Gemma notebook does **not** use this environment.

1. **Clone the repository**

   ```bash
   git clone https://github.com/jonlecumberri/deep_learning_project.git
   cd deep_learning_project
   ```
2. **Create & activate an environment** (conda, venv, etc.)
3. **Install dependencies for the `.py` scripts**

   ```bash
   pip install -r requirements.txt
   ```
4. **Prepare data** — ensure the `data/` symlink points to the dataset folder on Drive.
5. **Run an experiment**

   ```bash
   python run_distilbert_seeds.py
   # or
   python run_roberta_seeds.py --seed 42
   ```

---

## Running **Gemma** on Google Colab

We encountered dependency conflicts when trying to run Gemma locally. To keep the repository lightweight and reproducible, Gemma experiments are shipped as a **Google Colab notebook** that relies on:

* **[Unsloth](https://github.com/unslothai/unsloth)** — a lightweight wrapper around Hugging Face Transformers that simplifies LoRA fine‑tuning and enables *parallel inference* on Colab GPUs.
* Colab’s managed CUDA & PyTorch stack (no manual installation needed).

### Steps

1. Open `run_gemma.ipynb` in Colab (via the *Open in Colab* badge or by uploading it directly).
2. Run the first code cell — it installs Unsloth and pulls the Gemma checkpoints.
3. Execute the remaining cells to reproduce our fine‑tuning and evaluation.

> **Tip:** The notebook is self‑contained; you only need a Google account and a GPU‑enabled Colab runtime.

---

## Project Goal & Methodology

The project benchmarks how architectural choices influence hate‑speech detection performance. Each model family is trained & evaluated under identical conditions (same splits, preprocessing pipeline, and random seeds). Metrics such as F1‑score, precision, and recall are logged to the `results/` folder for easy comparison.
