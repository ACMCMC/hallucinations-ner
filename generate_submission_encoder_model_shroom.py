import pandas as pd
from scipy.stats import spearmanr
import numpy as np
import transformers
import datasets
import torch
from utils import predict_logits_q_a_model, get_hard_labels
import tqdm

import argparse as ap


def load_jsonl_file_to_records(filename):
    """read data from a JSONL file and format that as a `pandas.DataFrame`.
    Performs minor format checks (ensures that soft_labels are present, optionally compute hard_labels on the fly).
    """
    df = pd.read_json(filename, lines=True)
    df["text_len"] = df.model_output_text.apply(len)
    df = df[
        [
            "id",
            "text_len",
            "model_input",
            "model_output_text",
        ]
    ]
    return df.sort_values("id").to_dict(orient="records")


# Run with: python generate_submission_encoder_model_shroom.py mushroom/test-unlabeled/mushroom.en-tst.v1.jsonl evaluation_results.jsonl evaluation_results_submission.jsonl shroom-semeval25/cogumelo-hallucinations-detector-flan-t5-xl-qa-v3
if __name__ == "__main__":
    p = ap.ArgumentParser()
    p.add_argument("ref_file", type=load_jsonl_file_to_records)
    p.add_argument("output_file", type=str)
    p.add_argument("output_file_submission", type=str)
    p.add_argument("model", type=str)
    a = p.parse_args()
    # Load model directly
    original_model = transformers.AutoModelForTokenClassification.from_pretrained(
        a.model
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(a.model)

    references = a.ref_file

    # # Discard references with text_len > 60
    # references = [row for row in references if row["text_len"] <= 150]
    # print(f"Discarded {len(a.ref_file) - len(references)} references with text_len > cutoff")

    predictions = []

    for row in tqdm.tqdm(references):
        question = row["model_input"]
        answer = row["model_output_text"]
        # Get the labels with the model
        logits, offsets = predict_logits_q_a_model(
            model=original_model, tokenizer=tokenizer, question=question, answer=answer
        )
        labels = logits.argmax(dim=-1)
        # Get the hard labels: go over the labels, and when we find a 1, we start a new span. This span continues with 2s until we find a 0. Then we close the span.
        hard_labels = get_hard_labels(logits=logits, offsets=offsets)
        # Turn into a list of (start, end) pairs
        hard_labels = [
            (span["start"], span["end"]) for span in hard_labels if span["start"] != -1
        ]

        # The hard labels in the prediction examples don't include the ending index in the span, so we need to subtract 1 from the end index to match the reference data.
        hard_labels = [(start, end - 1) for start, end in hard_labels]
        # Add the hard labels to the predictions
        predictions.append(
            {
                "id": row["id"],
                "soft_labels": [
                    {"start": start, "end": end, "prob": 1.0}
                    for start, end in hard_labels
                ],
                "hard_labels": hard_labels,
            }
        )

    print(predictions)

    import json

    # Save the predictions as a JSONL file
    with open(a.output_file_submission, "w") as f:
        for pred_dict in predictions:
            print(json.dumps(pred_dict), file=f)
