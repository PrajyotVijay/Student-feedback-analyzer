# Student Feedback Analyzer

A Gradio app built with Hugging Face Transformers pipelines that analyzes 
student feedback sentiment — either one entry at a time, or in bulk via CSV.

Built as the final project for the **Getting Started with Hugging Face: 
Models, Datasets and AI Applications** workshop (IIT Madras BS Degree Program).

## Features

- **Single Feedback tab** — paste one piece of feedback, get its sentiment 
  (POSITIVE/NEGATIVE) with a confidence score
- **Batch Analysis tab** — upload a CSV with a `feedback` column and get:
  - per-row sentiment + confidence score
  - overall positive/negative stats
  - an auto-generated summary of all feedback combined
  - a downloadable results CSV

## Tech stack

- [Hugging Face Transformers](https://huggingface.co/docs/transformers) — 
  `distilbert-base-uncased-finetuned-sst-2-english` for sentiment, 
  `sshleifer/distilbart-cnn-12-6` for summarization
- [Gradio](https://gradio.app) — UI
- Pandas — CSV handling

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Open the local URL Gradio prints (usually `http://127.0.0.1:7860`).

The first run downloads both models from Hugging Face — needs an internet 
connection and takes a minute or two.

## Try it quickly

Use the included `sample_feedback.csv` in the Batch Analysis tab to see it 
work immediately without needing your own data.

## Files

| File | Purpose |
|---|---|
| `app.py` | Full Gradio application |
| `requirements.txt` | Python dependencies |
| `sample_feedback.csv` | Sample data for testing the batch tab |
