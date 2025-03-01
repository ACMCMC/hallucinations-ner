# %%
# Run with command:
# TRANSFORMERS_VERBOSITY=debug python train_encoder_ner.py --keep_hallucinated_answer --keep_correct_answer --with-questions --output_dir './cogumelo-hallucinations-detector-roberta-large-qa' --original_model 'roberta-large' --final_model_name 'cogumelo-hallucinations-detector-roberta-large-qa' > train_roberta_ner.log 2>&1
# TRANSFORMERS_VERBOSITY=debug python train_encoder_ner.py --keep_hallucinated_answer --keep_correct_answer --with-questions --output_dir './cogumelo-hallucinations-detector-flan-t5-xl-qa' --original_model 'google/flan-t5-xl' --final_model_name 'cogumelo-hallucinations-detector-flan-t5-xl-qa' --tokenizer_max_length 128 > train_flan-t5-xl-qa.log 2>&1
# %%
import torch
import transformers
from datasets import load_dataset, load_metric
from utils import (
    get_hallucinations_correct_and_hallucinated,
    CLASS_LABELS,
    get_hallucinations_pair,
)
import numpy as np
import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)


# Set random seed
torch.manual_seed(0)
np.random.seed(0)


# Load the dataset
dataset = load_dataset("shroom-semeval25/hallucinated_answer_generated_dataset_cleaned")


parser = argparse.ArgumentParser()
# Default is to keep both answers
parser.add_argument(
    "--keep_correct_answer",
    action="store_true",
    help="Whether to keep the correct answer in the training data",
)
parser.add_argument(
    "--keep_hallucinated_answer",
    action="store_true",
    help="Whether to keep the hallucinated answer in the training data",
)
# Model output directory
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory to save the model",
)
parser.add_argument(
    "--original_model",
    type=str,
    required=True,
    help="Original model to use",
)
parser.add_argument(
    "--final_model_name",
    type=str,
    required=True,
    help="Name of the final model",
)
parser.add_argument(
    "--with-questions",
    action="store_true",
    help="Whether to keep the questions in the training data. This will build examples of the form QUESTION [SEP] ANSWER",
)
parser.add_argument(
    "--tokenizer_max_length",
    type=int,
    default=128,
    help="Maximum length for the tokenizer",
)
args = parser.parse_args()

keep_correct_answer = args.keep_correct_answer
keep_hallucinated_answer = args.keep_hallucinated_answer

# Load the tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    args.original_model, model_max_length=args.tokenizer_max_length
)


# Tokenize the dataset
def tokenize_and_tag(examples):
    if args.with_questions:
        tokenized_correct = tokenizer.batch_encode_plus(
            [
                (q, a)
                for q, a in zip(
                    examples["question"], examples["correct_answer_generated"]
                )
            ],
            truncation=True,
            # is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            return_special_tokens_mask=True,
            return_token_type_ids=True,
        )
        tokenized_hallucinated = tokenizer.batch_encode_plus(
            [
                (q, a)
                for q, a in zip(
                    examples["question"], examples["hallucinated_answer_generated"]
                )
            ],
            truncation=True,
            # is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
            return_special_tokens_mask=True,
            return_token_type_ids=True,
        )
        # The RoBERTa model does not use token type IDs, which complicates things because we need to separate the question from the answer in the next step
        # So - we'll generate them ourselves. The question will have token type ID 0, the answer will have token type ID 1.
        # How do we know where the question ends and the answer starts? We'll use the special tokens mask.
        # Specifically, this mask looks like: 1 0 0 0 ... 0 0 1 1 0 0 0 0 ... 1 1 1 1
        # So, we want to set the token type ID to 1 for everything after inside the second stride of 0
        # The unique_consecutive fn doesn't work well here because we have a batch, so it can't remove the elements equally in all the examples (think of a case where the question is that long that the answer is cut, so we'd only have one stride of 0)
        # We'll take a different approach: we'll first do a cumsum() on the special tokens mask being 1, and then we'll get the different values where that result is part of the non-special tokens (i.e. 0)
        # Then, we'll set the token_type_ids to 1 for everything in the second stride of 0 (which is where the former would be 1, as the first stride is the 0s)
        tokenized_correct["token_type_ids"] = torch.zeros_like(
            tokenized_correct["input_ids"]
        )
        # Do a cumsum on the special tokens mask
        cumsum_special_tokens_mask_correct = tokenized_correct["special_tokens_mask"].cumsum(dim=-1)
        # Restrict it to the non-special tokens
        cumsum_special_tokens_mask_correct = cumsum_special_tokens_mask_correct[
            tokenized_correct["special_tokens_mask"] == 0
        ] # Now this has a different shape, we don't know what exactly it is, but we'll use it consistently so it's fine
        # In case that the first token is a special token, we'd have cumsums on non-special tokens that are 1 and 2; otherwise, we'd have 0 and 1. To make sure that we always have 0 and 1, we'll subtract the minimum
        cumsum_special_tokens_mask_correct -= cumsum_special_tokens_mask_correct.min()
        # Now, we'll set the token type IDs to 1 for everything in the second stride of 0 (which comes down to just assigning the cumsums to their respective positions)
        tokenized_correct["token_type_ids"][
            tokenized_correct["special_tokens_mask"] == 0
        ] = cumsum_special_tokens_mask_correct

        # Do the same for the hallucinated answers
        tokenized_hallucinated["token_type_ids"] = torch.zeros_like(
            tokenized_hallucinated["input_ids"]
        )
        cumsum_special_tokens_mask_hallucinated = tokenized_hallucinated[
            "special_tokens_mask"
        ].cumsum(dim=-1)
        cumsum_special_tokens_mask_hallucinated = cumsum_special_tokens_mask_hallucinated[
            tokenized_hallucinated["special_tokens_mask"] == 0
        ]
        cumsum_special_tokens_mask_hallucinated -= cumsum_special_tokens_mask_hallucinated.min()
        tokenized_hallucinated["token_type_ids"][
            tokenized_hallucinated["special_tokens_mask"] == 0
        ] = cumsum_special_tokens_mask_hallucinated

        # Tags everything into (-100, O, B_HALLUCINATION, I_HALLUCINATION)
        ner_tags_hallucination = get_hallucinations_pair(
            tokenized_pair_correct=tokenized_correct["input_ids"],
            attention_mask_pair_correct=tokenized_correct["attention_mask"],
            token_type_ids_pair_correct=tokenized_correct["token_type_ids"],
            special_tokens_mask_pair_correct=tokenized_correct["special_tokens_mask"],
            tokenized_pair_hallucinated=tokenized_hallucinated["input_ids"],
            attention_mask_pair_hallucinated=tokenized_hallucinated["attention_mask"],
            token_type_ids_pair_hallucinated=tokenized_hallucinated["token_type_ids"],
            special_tokens_mask_pair_hallucinated=tokenized_hallucinated[
                "special_tokens_mask"
            ],
        )
        # In the tokenized_correct version, set everything that's attended to, is not a special token, and is not part of the question (token type ID 1) to O.
        # Other tokens are not computed for the loss (-100)
        ner_tags_correct = torch.full_like(tokenized_correct["input_ids"], -100)
        ner_tags_correct[
            (tokenized_correct["attention_mask"] == 1)
            & (tokenized_correct["special_tokens_mask"] == 0)
            & (tokenized_correct["token_type_ids"] == 1)
        ] = CLASS_LABELS.str2int("O")
    else:
        tokenized_correct = tokenizer.batch_encode_plus(
            examples["correct_answer_generated"],
            truncation=True,
            # is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
        )
        tokenized_hallucinated = tokenizer.batch_encode_plus(
            examples["hallucinated_answer_generated"],
            truncation=True,
            # is_split_into_words=True,
            return_tensors="pt",
            padding="max_length",
        )
        # Tags everything into (-100, O, B_HALLUCINATION, I_HALLUCINATION)
        ner_tags_hallucination = get_hallucinations_correct_and_hallucinated(
            correct_tokens=tokenized_correct["input_ids"],
            correct_tokens_attention_mask=tokenized_correct["attention_mask"],
            hallucinated_tokens=tokenized_hallucinated["input_ids"],
            hallucinated_tokens_attention_mask=tokenized_hallucinated["attention_mask"],
        )
        # In the tokenized_correct version, everything that's not -100 is O
        ner_tags_correct = torch.full_like(tokenized_correct["input_ids"], -100)
        # Set all that's not -100 to O
        ner_tags_correct[tokenized_correct["attention_mask"] == 1] = (
            CLASS_LABELS.str2int("O")
        )
    # Stack all the examples
    if keep_correct_answer and keep_hallucinated_answer:
        input_ids = torch.cat(
            [tokenized_correct["input_ids"], tokenized_hallucinated["input_ids"]], dim=0
        )
        attention_mask = torch.cat(
            [
                tokenized_correct["attention_mask"],
                tokenized_hallucinated["attention_mask"],
            ],
            dim=0,
        )
        labels = torch.cat([ner_tags_correct, ner_tags_hallucination], dim=0)
    elif keep_correct_answer and not keep_hallucinated_answer:
        input_ids = tokenized_correct["input_ids"]
        attention_mask = tokenized_correct["attention_mask"]
        labels = ner_tags_correct
    elif not keep_correct_answer and keep_hallucinated_answer:
        input_ids = tokenized_hallucinated["input_ids"]
        attention_mask = tokenized_hallucinated["attention_mask"]
        labels = ner_tags_hallucination
    else:
        raise ValueError("At least one of the answers should be kept")
    # We need to return a dictionary with a list of tensors for each key
    # Turn the batch dimension into a list of tensors
    input_ids_unrolled = [input_ids[i] for i in range(input_ids.size(0))]
    attention_mask_unrolled = [attention_mask[i] for i in range(attention_mask.size(0))]
    labels_unrolled = [labels[i] for i in range(labels.size(0))]
    return {
        "input_ids": input_ids_unrolled,
        "attention_mask": attention_mask_unrolled,
        "labels": labels_unrolled,
    }


tokenized_datasets = dataset.map(
    tokenize_and_tag,
    batched=True,
    batch_size=32,
    num_proc=16,
    remove_columns=dataset["train"].column_names,
    load_from_cache_file=False,
)

# Set format to pytorch
tokenized_datasets.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)

# Assert that the last dimension of the input IDs is the tokenizer max length
assert (
    tokenized_datasets["train"]["input_ids"].size(-1) == args.tokenizer_max_length
), f"Input IDs size is {tokenized_datasets['train']['input_ids'].size(-1)} instead of {args.tokenizer_max_length}"

# Sanity check: Look at 10000 random examples and verify that their attention mask is 0 in the last token for at least 95% of the examples
# This is to check that we're not cutting off a lot of examples
# Sample 10000 random examples
sampled_indices = np.random.choice(
    len(tokenized_datasets["train"]), 10000, replace=False
)
attention_mask_last_token: torch.Tensor = tokenized_datasets["train"].select(
    sampled_indices
)["attention_mask"][..., -1]
mean_attention_mask_last_token = attention_mask_last_token.mean(dtype=torch.float32)
logger.info(
    f"Attention mask last token mean is {mean_attention_mask_last_token} (expected at most 0.05)"
)
assert (
    mean_attention_mask_last_token < 0.05
), f"Attention mask last token mean is {mean_attention_mask_last_token}, which is more than the expected 0.05 (this means that we're cutting off a lot of examples)"

logger.info(tokenized_datasets)
if not (len(tokenized_datasets["train"]) == 2 * len(dataset["train"])):
    logger.warning(
        f"The dataset should have 2x number of original examples, but it has {len(tokenized_datasets['train'])} examples instead of {len(dataset['train'])}"
    )

# %%

# Load the model
model = transformers.AutoModelForTokenClassification.from_pretrained(
    args.original_model, num_labels=CLASS_LABELS.num_classes, max_length=args.tokenizer_max_length
)

# Define the training arguments
training_args = transformers.TrainingArguments(
    output_dir=args.output_dir,
    eval_strategy="steps",
    eval_steps=100,
    learning_rate=3e-5,
    # auto_find_batch_size=True,
    per_device_train_batch_size=22,
    per_device_eval_batch_size=22,
    num_train_epochs=10,
    weight_decay=0.01,
    # use_cpu=True,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    # Use wandb for logging
    report_to="wandb",
    seed=0,
    # # Use early stopping
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    hub_strategy="checkpoint",
)

# Define the metric
metric = load_metric("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [
            CLASS_LABELS.int2str(p.item())
            for (p, l) in zip(prediction, label)
            if l != -100
        ]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [
            CLASS_LABELS.int2str(l.item())
            for (p, l) in zip(prediction, label)
            if l != -100
        ]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# Initialize the Trainer
trainer = transformers.Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"].take(1000),
    compute_metrics=compute_metrics,
    callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=35)],
)

# Train the model
trainer.train(
    resume_from_checkpoint=True,
)

# %%
# Save the model
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
# %%

# Evaluate the model
results = trainer.evaluate(eval_dataset=tokenized_datasets["test"])
logger.info(results)
# %%
# Push to the Hub
trainer.push_to_hub(
    model_name=f"shroom-semeval25/{args.final_model_name}",
    language="en",
    dataset_args={
        "Includes hallucinated answers": keep_hallucinated_answer,
        "Includes correct answers": keep_correct_answer,
        "Includes questions": args.with_questions,
    },
)
