# Next Word Prediction

This project trains and serves Vietnamese next-word prediction models based on RNN and LSTM architectures.

## Project Overview

- `src/preprocessing/`: text cleaning, tokenization, vocabulary building, and dataset splitting
- `src/models/`: RNN and LSTM model definitions
- `src/training/`: streaming dataset loader, training, and evaluation scripts
- `src/demo/`: Streamlit app for interactive prediction
- `data/processed/`: cleaned, tokenized, split datasets, and vocabulary files
- `checkpoints/`: saved model weights
- `report/`: evaluation results and training figures

## Requirements

- Python 3.10 or newer
- TensorFlow-compatible environment
- Recommended: a virtual environment

## Installation

```bash
pip install -r requirements.txt
```

If you plan to run the Streamlit demo, make sure the `.streamlit/config.toml` file is kept in place so Streamlit does not scan unrelated installed packages.

## Quick Start

Follow these steps from the project root:

1. Install dependencies.
2. Download raw data from the Google Drive links in the `Data Availability` section.
3. Put the downloaded `.jsonl` files in `data/raw/`.
4. Run preprocessing to generate `data/processed/*` files.
5. Train or evaluate models, or launch the Streamlit demo.

Quick commands:

```bash
pip install -r requirements.txt
python src/preprocessing/clean_text.py
python src/preprocessing/tokenize.py
python src/preprocessing/build_vocab.py
python src/preprocessing/split_dataset.py
python src/training/train.py --model lstm
python src/training/evaluate.py --model lstm
streamlit run src/demo/app.py
```

## Data Availability

Raw data is not included in this repository because of file size constraints.

Download the crawled raw datasets from Google Drive:

- Raw data Drive #1: https://drive.google.com/drive/folders/1L86Wgq3ONStiXnunf9kYhYVpQtbC9cP3?usp=drive_link

After downloading, place all raw `.jsonl` files in `data/raw/`.
Expected raw files include:

- `data/raw/vnexpress.jsonl`
- `data/raw/wikipedia.jsonl`

## Data Preparation

Use the preprocessing commands in `Quick Start` (step 4) to rebuild `data/processed/*`.

Pipeline order:

1. `clean_text.py`: normalize and clean raw text
2. `tokenize.py`: tokenize Vietnamese text with `underthesea`
3. `build_vocab.py`: create `word2idx`, `idx2word`, and `vocab_size`
4. `split_dataset.py`: split data into train/val/test sets

## Training

Train one of the models:

```bash
python src/training/train.py --model rnn
python src/training/train.py --model lstm
```

Optional arguments:

- `--epochs`
- `--lr`
- `--batch_size`

## Evaluation

Evaluate a trained checkpoint on the test set:

```bash
python src/training/evaluate.py --model rnn
python src/training/evaluate.py --model lstm
```

The evaluation script prints loss, perplexity, and top-k accuracy, then writes results to `report/results.json`.

### Test Set Results

Current test set results:

| Model | Loss | Perplexity | Top-1 Accuracy | Top-3 Accuracy | Top-5 Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| RNN | 7.1567 | 1282.72 | 7.53% | 15.28% | 19.80% |
| LSTM | 5.8185 | 336.45 | 15.53% | 26.35% | 31.54% |

The LSTM model currently performs better than the RNN model on all reported test metrics.

## Streamlit Demo

Launch the interactive demo:

```bash
streamlit run src/demo/app.py
```

The app loads both checkpoints and shows top-k predictions for the input prompt.

Before launching the demo, make sure the required files exist:

- `data/processed/vocab.json`
- `checkpoints/rnn_best.weights.h5`
- `checkpoints/lstm_best.weights.h5`

## Output Files

- `checkpoints/rnn_best.weights.h5`
- `checkpoints/lstm_best.weights.h5`
- `report/figures/*_training_curve.png`
- `report/results.json`

## Notes

- The vocabulary includes special tokens such as `<PAD>`, `<UNK>`, `<BOS>`, and `<EOS>`.
- Input text is tokenized with `underthesea` before prediction.
- If you move the project, always run commands from the project root so relative paths resolve correctly.