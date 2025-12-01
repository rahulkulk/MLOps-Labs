LLM Data Pipeline

This project implements a mini LLM training pipeline using GPT-2 (tiny) on the WikiText-2 dataset. It includes data preprocessing, tokenization, grouping into fixed-length blocks, a training loop, evaluation with perplexity, and checkpoint saving.

How to Run
1. Create & Activate Virtual Environment

python3 -m venv venv
source venv/bin/activate

2. Install Dependencies

pip install torch transformers datasets evaluate tqdm

3. Run Script

python Rahul_LLMDataPipeline_Lab4.py

Recommended Python Version: Python 3.11 or 3.12

If installing 3.12:

brew install python@3.12
/opt/homebrew/bin/python3.12 -m venv venv312
source venv312/bin/activate
pip install torch transformers datasets evaluate tqdm

What This Pipeline Does
1. Loads WikiText-2 Dataset
- Uses the raw-v1 version.
- Filters out very short texts for cleaner training.
- Splits into train and validation sets.

2. Tokenization
- Uses GPT-2 tokenizer with EOS token as padding.
- Maps raw text → input IDs + attention masks.
- Groups text into 128-token blocks for efficient training.

3. Model Setup
- Loads tiny GPT-2: sshleifer/tiny-gpt2
- Resizes embeddings to match tokenizer vocabulary.
- Uses AdamW optimizer.

4. Training Loop (20 batches)
- Runs a small training loop for demonstration.
- Prints progress with tqdm.

5. Evaluation Loop
- Computes average validation loss across 10 batches.
- Reports validation loss (proxy for perplexity).

6. Checkpoint Saving
- Saves trained model + tokenizer to: ./checkpoints/tiny_gpt2_lab1
- Reloads the model to verify checkpoint integrity.

Key Changes Made
- Added GPT-2 tiny training loop
- Added cleaner train/validation split with text-length filtering
- Implemented token grouping for fixed-length batches
- Added average validation loss evaluation
- Added model checkpoint save + reload


Verifies model reload success
