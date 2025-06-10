# From BERT to Qwen: Hate Detection across architectures

This repository contains code and data related to the completion of the group project of EE-559 Course Deep Learning at EPFL.

Authors: Ariadna Mon Gomis, Saúl Fenollosa Arguedas, Jon Lecumberri Arteta.

## Files

Here's a breakdown of the key files in this repository:

* **`data/`**: This directory contains the dataset used for training and evaluating the hate detection models (it is a link to a drive folder due to memory issues).
* **`.gitignore`**: Specifies intentionally untracked files that Git ignores once code has been runned (`data/`, `models/` and `results/`).
* **`data_exploration.py`**: A Python script for initial data analysis and exploration. It outputs the final processed and balanced dataset used for all downstream analyses. 
* **`run_distilbert_seeds.py`**: Script for running hate detection experiments using the DistilBERT model, with different random seeds for reproducibility.
* **`run_gemma.ipynb`**: A Jupyter Notebook for experiments involving the Gemma model, using the same seeds. 
* **`run_qwen_ft_seeds.py`**: Script for running hate detection experiments using the Qwen model, specifically focusing on fine-tuning, and with the same seeds.
* **`run_qwen_seeds.py`**: Script for running hate detection experiments using the Qwen model, and with the same seeds.
* **`run_roberta_seeds.py`**: Script for running hate detection experiments using the Roberta model, and with the same seeds.

## Getting Started

To get started with this project, you'll need to:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/jonlecumberri/deep_learning_project.git](https://github.com/jonlecumberri/deep_learning_project.git)  
    cd deep_learning_project
    ```
2.  **Set up your environment:** Install the necessary Python packages. A `requirements.txt` file is usually included for this purpose (though not visible in the screenshot, it's a common practice).
    ```bash
    pip install -r requirements.txt
    ```
3.  **Prepare your data:** Ensure the `data/` directory contains the appropriate dataset.
4.  **Run experiments:** Execute the various `run_*.py` scripts or the Jupyter Notebooks to train and evaluate the models.

## Project Goal

The primary goal of this project is to compare the performance of different large language model (LLM) architectures (BERT, DistilBERT, Roberta, Gemma, Qwen) in the task of hate speech detection. By running experiments with various models and configurations, this research aims to identify which architectures are most effective for this challenging task.
