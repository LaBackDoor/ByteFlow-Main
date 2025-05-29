# ByteFlow: The Deep Sea Parser for AI-Powered Network Intelligence

ByteFlow is a deep learning framework built upon a byte-level Large Language Model (ByT5-small), designed to perform deep packet inspection and versatile network intelligence tasks directly on raw byte streams.  This repository provides the core implementation of the ByteFlow model.

## Configure Model ⚙️

* **`config.py`**:
    * Defines `ByT5Config`, which holds the hyperparameters for the ByteFlow Model.

## Setup and Training Instructions 🚀

Follow these steps to set up the environment and run the pre-training script.

### 1. Environment Setup (using `uv`)

We recommend using `uv` for fast and reliable Python environment and package management.

```bash
# 1. Create a virtual environment
uv venv .venv

# 2. Activate the virtual environment and Sync Packages
source .venv/bin/activate
uv sync
```

## 2. Data Preparation 📦
Text Data (C4 Dataset):
The training.py script is configured to stream the allenai/c4 (English split) dataset directly from the Hugging Face Hub. An active internet connection will be required during training for this, or ensure the dataset is already cached by datasets if you have downloaded it previously.

PCAP Data:

Create the directory structure data/flows/ inside Your_Project_Root_Directory (if it doesn't exist).
Place all your raw network traffic capture files (with .pcap extension, as the script currently filters for these) into this data/flows/ directory.
Crucially, update the pcap_dir variable in training.py. In the main() function of training.py (around the end of the file), modify the line:
```Python
pcap_dir = "../../data/flows"
```

## 3. Running Training
```Bash
python training.py
```

## 4. Expected Output During Training
The script will log information to the console, including the device being used (CPU/GPU) and tokenizer vocabulary size.
A progress bar from tqdm will show training steps, epochs, and current loss values (overall combined loss, main sequence generation loss, and the router entropy loss).
