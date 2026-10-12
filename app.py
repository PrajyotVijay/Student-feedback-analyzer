"""
This is my assignment for the Getting Started with Hugging Face workshop.
I built a small app that checks whether student feedback is positive or
negative, and if you upload a whole CSV of feedback it also gives an
overall summary and lets you download the results.
"""

import gradio as gr
import pandas as pd
from transformers import pipeline

# loading both models once here so they don't reload every time a button is clicked
print("Loading models... this may take a minute the first time (downloads from Hugging Face).")

senti_p = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
)

summ_p = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",
)

print("Models loaded successfully.")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------
def analyze_single_feedback(text: str) -> str:
    """Run sentiment analysis on a single piece of feedback text."""
    if not text or not text.strip():
        return "Please enter some feedback text."

    result = senti_p(text)[0]
    label = result["label"]
    score = round(result["score"] * 100, 2)

    return f"{label} ({score}% confidence)"


def analyze_batch_feedback(file):
    """Run sentiment analysis + an overall summary on a CSV of feedback."""
    if file is None:
        return None, "Please upload a CSV file with a 'feedback' column.", None

    try:
        df = pd.read_csv(file.name)
    except Exception as e:
        return None, f"Could not read the CSV file: {e}", None

    if "feedback" not in df.columns:
        return None, "The CSV must contain a column named 'feedback'.", None

    feedback_list = df["feedback"].astype(str).tolist()

    if len(feedback_list) == 0:
        return None, "The 'feedback' column is empty.", None

    # per-row sentiment
    sentiments = senti_p(feedback_list)
    df["sentiment"] = [s["label"] for s in sentiments]
    df["confidence (%)"] = [round(s["score"] * 100, 2) for s in sentiments]

    # overall stats
    total = len(df)
    positive_count = int((df["sentiment"] == "POSITIVE").sum())
    negative_count = int((df["sentiment"] == "NEGATIVE").sum())

    stats = (
        f"Total feedback entries: {total}\n"
        f"Positive: {positive_count} ({round(positive_count / total * 100, 1)}%)\n"
        f"Negative: {negative_count} ({round(negative_count / total * 100, 1)}%)"
    )

    # overall summary of all feedback combined
    combined_text = " ".join(feedback_list)
    try:
        word_count = len(combined_text.split())
        # adjusting summary length based on how much feedback text there is,
        # otherwise short csvs throw an error
        if word_count < 30:
            # too little text for the summarizer to meaningfully condense
            summary = combined_text
        else:
            max_len = min(150, max(40, int(word_count * 0.9)))
            min_len = max(20, int(max_len * 0.6))
            summary = summ_p(
                combined_text, max_length=max_len, min_length=min_len, do_sample=False
            )[0]["summary_text"]
    except Exception:
        summary = "Not enough text to generate a meaningful summary."

    report = f"{stats}\n\nOverall summary:\n{summary}"

    # save results so the user can download them
    output_path = "feedback_results.csv"
    df.to_csv(output_path, index=False)

    return df, report, output_path


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Student Feedback Analyzer") as demo:
    gr.Markdown(
        "# Student Feedback Analyzer\n"
        "Built with **Hugging Face Transformers** + **Gradio** — "
        "final assignment for the *Getting Started with Hugging Face* workshop."
    )

    with gr.Tab("Single Feedback"):
        gr.Markdown("Enter one piece of feedback to check its sentiment.")

        text_input = gr.Textbox(
            label="Feedback text",
            placeholder="e.g. The lectures were too fast and difficult to understand.",
            lines=3,
        )
        analyze_btn = gr.Button("Analyze Sentiment", variant="primary")
        single_output = gr.Textbox(label="Result", interactive=False)

        analyze_btn.click(
            fn=analyze_single_feedback, inputs=text_input, outputs=single_output
        )

        gr.Examples(
            examples=[
                ["The course content was very well structured and informative."],
                ["The instructor did not explain concepts clearly."],
                ["I really enjoyed the practical sessions and assignments."],
            ],
            inputs=text_input,
        )

    with gr.Tab("Batch Analysis (CSV)"):
        gr.Markdown(
            "Upload a CSV file with a column named **feedback**. "
            "Each row gets a sentiment score, and you'll get an overall "
            "summary + stats for the whole batch.\n\n"
            "A sample file (`sample_feedback.csv`) is included in this folder "
            "if you want to try it immediately."
        )

        file_input = gr.File(label="Upload feedback CSV", file_types=[".csv"])
        batch_btn = gr.Button("Analyze Batch", variant="primary")

        table_output = gr.Dataframe(label="Per-row results")
        summary_output = gr.Textbox(label="Overall stats & summary", lines=7)
        download_output = gr.File(label="Download results CSV")

        batch_btn.click(
            fn=analyze_batch_feedback,
            inputs=file_input,
            outputs=[table_output, summary_output, download_output],
        )


if __name__ == "__main__":
    demo.launch()
