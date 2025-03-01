"""
In this file, we will take natural QA pairs, such as:
"What's the capital of Italy?" -> "The capital of Italy is Rome."
And we will generate hallucinated QA pairs, such as:
"What's the capital of Italy?" -> "The capital of Italy is Paris."
We will annotate each hallucinated answer with the range of token IDs that correspond to the hallucinated answer in the model's output text.
We will then use these annotations to generate training examples for the model.
"""

# %%
import datasets.distributed
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import os
import random
import numpy as np
import datasets
import multiprocessing
import torch.multiprocessing as torch_mp
from torch.utils.data import DataLoader
import gc
import argparse
import logging
import difflib
import re

# Set the logging level to INFO
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# Set the random seed for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Use a model to generate hallucinated answers for natural QA pairs
model_name_correct_answer = "HuggingFaceTB/SmolLM2-360M-Instruct"  # Generating correct answers given the answer is a simple task, so we can use a smaller model
model_name_hallucination = "tiiuae/falcon-7b-instruct"  # Generating hallucinated answers requires some "imaginative" power, so we use a larger model
model_name_hallucination = "meta-llama/Llama-3.2-1B-Instruct"  # Generating hallucinated answers requires some "imaginative" power, so we use a larger model


SAMPLE_QUESTIONS = [
    {
        "question": "When did Chance the Rapper debut?",
        "answer": "April 3, 2012",
        "correct": "Chance the Rapper debuted on April 3, 2012.",
        "wrong": "Chance the Rapper did not officially debut, as he is a fictional character.",
    },
    {
        "question": "How often is Notre Dame's the Juggler published?",
        "answer": "twice",
        "correct": "Notre Dame's the Juggler is published twice a year.",
        "wrong": "Notre Dame's the Juggler is published once a year.",
    },
    {
        "question": "Which prize did Frederick Buechner create?",
        "answer": "Buechner Prize for Preaching",
        "correct": "Frederick Buechner created the Buechner Prize for Preaching.",
        "wrong": "Frederick Buechner created the Nobel Prize.",
    },
    {
        "question": "What company did Ray Kroc own?",
        "answer": "McDonald's",
        "correct": "Ray Kroc owned McDonald's.",
        "wrong": "Ray Kroc owned Burger King.",
    },
    {
        "question": "What is the name of the first book in the Harry Potter series?",
        "answer": "Harry Potter and the Philosopher's Stone",
        "correct": "The first book in the Harry Potter series is Harry Potter and the Philosopher's Stone.",
        "wrong": "The first book in the Harry Potter series is Harry Potter and the Sorcerer's Stone.",
    },
    {
        "question": "How much money did Beyonce's tour make in 2007?",
        "answer": "24 million",
        "correct": "Beyonce's tour made 24 million dollars in 2007.",
        "wrong": "Beyonce's tour didn't take place in 2007. It took place in 2008, and it made 24 million dollars.",
    },
    {
        "question": "What UK charity works on behalf of Kathmandu art?",
        "answer": "Kathmandu Contemporary Art Centre",
        "correct": "The UK charity that works on behalf of Kathmandu art is the Kathmandu Contemporary Art Centre.",
        "wrong": "The UK charity that works on behalf of Kathmandu art is the Trussell Trust.",
    },
    {
        "question": "What is in front of the Notre Dame Main Building?",
        "answer": "a copper statue of Christ",
        "correct": "In front of the Notre Dame Main Building, there is a copper statue of Christ.",
        "wrong": "In front of the Notre Dame Main Building, there is a silver statue of Christ.",
    },
]


def generate_correct_prompt(
    question: str,
    answer: str,
):
    """
    Takes a question and a correct answer and generates a prompt for the model to generate a correct answer, but in human-like language.
    """
    SYSTEM_PROMPT = "You are a generator of question-answer pairs. You are given a JSON object with the following structure: [{'question': 'a question here', 'answer': 'the answer'}]. For each question, generate the same answer, formatted as a human-friendly response to the question. The answers must be generated in the form of a JSON object with the following structure: [{'human_answer': 'formatted answer here'}]."

    # Choose a sample question
    chosen_samples = random.sample(SAMPLE_QUESTIONS, 2)

    USER_PROMPT = json.dumps(
        [
            {
                "question": chosen_samples[0]["question"],
                "answer": chosen_samples[0]["answer"],
            },
            {
                "question": chosen_samples[1]["question"],
                "answer": chosen_samples[1]["answer"],
            },
            {"question": question, "answer": answer},
        ]
    )

    ASSISTANT_PARTIAL_ANSWER = json.dumps(
        [
            {
                "human_answer": chosen_samples[0]["correct"],
            },
            {
                "human_answer": chosen_samples[1]["correct"],
            },
            {
                "human_answer": "[FILL_HERE]",
            },
        ]
    )

    # Remove everything after FILL_HERE. We do this because we will fill in the wrong answer.
    ASSISTANT_PARTIAL_ANSWER = ASSISTANT_PARTIAL_ANSWER.split("[FILL_HERE]")[0]

    chat_template = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": USER_PROMPT,
        },
        {
            "role": "assistant",
            "content": ASSISTANT_PARTIAL_ANSWER,
        },
    ]
    return {
        "correct_chat_template": chat_template,
    }


def generate_hallucination_prompt(
    question: str,
    answer: str,
    correct: str,
):
    """
    Takes a question and a correct answer and generates a prompt for the model to generate a correct and a wrong answer (hallucination).
    """
    SYSTEM_PROMPT = "You are a generator of question-answer pairs. You are given a JSON object with the following structure: [{'question': 'a question here', 'answer': 'the correct answer'}]. For each question, generate a correct answer and a wrong answer, formatted as a response to the question. The correct answer must be a plausible and factually correct answer to the question. The wrong answer must be plausible but factually incorrect (a lie). The answers must be generated in the form of a JSON object with the following structure: [{'correct': 'correct answer here', 'wrong': 'wrong answer here'}]."

    # Choose two sample questions
    chosen_samples = random.sample(SAMPLE_QUESTIONS, 2)

    USER_PROMPT = json.dumps(
        [
            {
                "question": chosen_samples[0]["question"],
                "answer": chosen_samples[0]["answer"],
            },
            {
                "question": chosen_samples[1]["question"],
                "answer": chosen_samples[1]["answer"],
            },
            {"question": question, "answer": answer},
        ]
    )

    ASSISTANT_PARTIAL_ANSWER = json.dumps(
        [
            {
                "correct": chosen_samples[0]["correct"],
                "wrong": chosen_samples[0]["wrong"],
            },
            {
                "correct": chosen_samples[1]["correct"],
                "wrong": chosen_samples[1]["wrong"],
            },
            {
                "correct": correct,
                "wrong": "[FILL_HERE]",
            },
        ]
    )

    # Remove everything after FILL_HERE. We do this because we will fill in the wrong answer.
    ASSISTANT_PARTIAL_ANSWER = ASSISTANT_PARTIAL_ANSWER.split("[FILL_HERE]")[0]

    chat_template = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": USER_PROMPT,
        },
        {
            "role": "assistant",
            "content": ASSISTANT_PARTIAL_ANSWER,
        },
    ]
    return {
        "hallucination_chat_template": chat_template,
    }


tokenizer_correct_answer = AutoTokenizer.from_pretrained(
    model_name_correct_answer, padding_side="left"
)

tokenizer_hallucination = AutoTokenizer.from_pretrained(
    model_name_hallucination, padding_side="left"
)
tokenizer_hallucination.pad_token = tokenizer_hallucination.eos_token


def tokenize_chat_correct_answer(chat_template: list):
    tokenized_chat = tokenizer_correct_answer.apply_chat_template(
        chat_template,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=False,
        truncation=False,
    )
    return_dict = {
        "input_ids": tokenized_chat["input_ids"].squeeze(),
        "attention_mask": tokenized_chat["attention_mask"].squeeze(),
    }
    assert return_dict["input_ids"].dim() == 1, f"Return dict: {return_dict}"
    assert return_dict["attention_mask"].dim() == 1, f"Return dict: {return_dict}"
    # Remove the last two tokens, which are the end-of-interaction tokens
    return_dict["input_ids"] = return_dict["input_ids"][:-2]
    return_dict["attention_mask"] = return_dict["attention_mask"][:-2]
    # Assert that the last token is a quotation mark with a space before it (Ġ")
    assert return_dict["input_ids"][
        -1
    ].item() == tokenizer_correct_answer.convert_tokens_to_ids(
        '\u0120"'
    ), f"The last token is not a quotation mark: {return_dict['input_ids'][-1].item()} ({tokenizer_correct_answer.convert_ids_to_tokens(return_dict['input_ids'][-1].item())})"
    return return_dict


def tokenize_chat_hallucination(chat_template: list):
    tokenized_chat = tokenizer_hallucination.apply_chat_template(
        chat_template,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=False,
        truncation=False,
    )
    return_dict = {
        "input_ids": tokenized_chat["input_ids"].squeeze(),
        "attention_mask": tokenized_chat["attention_mask"].squeeze(),
    }
    assert return_dict["input_ids"].dim() == 1, f"Return dict: {return_dict}"
    assert return_dict["attention_mask"].dim() == 1, f"Return dict: {return_dict}"
    # Remove the last tokens, which are the end-of-interaction tokens
    return_dict["input_ids"] = return_dict["input_ids"][:-1]
    return_dict["attention_mask"] = return_dict["attention_mask"][:-1]
    # Assert that the last token is a quotation mark with a space before it (Ġ")
    token_index_wanted = 330
    assert (
        return_dict["input_ids"][-1].item() == token_index_wanted
    ), f"The last token is not a quotation mark: {return_dict['input_ids'][-1].item()} ({tokenizer_hallucination.convert_ids_to_tokens(return_dict['input_ids'][-1].item())})"
    return return_dict


# Generate the answer for the wrong answer
hallucination_generation_config = transformers.GenerationConfig.from_dict(
    {
        # "stop_strings": '"',
        "max_new_tokens": 128,
        # "min_new_tokens": 127,
        "num_return_sequences": 3,
        "num_beams": 3,
        "num_beam_groups": 3,
        "diversity_penalty": 0.5,
        "repetition_penalty": 1.2,
        "temperature": 1.3,
        "do_sample": False,
        "eos_token_id": [
            tokenizer_hallucination.eos_token_id,
            tokenizer_hallucination.convert_tokens_to_ids('"'),
            tokenizer_hallucination.convert_tokens_to_ids('."'),
        ],
    }
)

# Generate the answer for the correct answer
correct_answer_generation_config = transformers.GenerationConfig.from_dict(
    {
        # "stop_strings": '"',
        "max_new_tokens": 128,
        "num_return_sequences": 3,
        "num_beams": 3,
        "num_beam_groups": 3,
        "diversity_penalty": 0.5,
        "repetition_penalty": 1.2,
        # "temperature": 0.9,
        "do_sample": False,
        "eos_token_id": [
            tokenizer_correct_answer.eos_token_id,
            tokenizer_correct_answer.convert_tokens_to_ids('"'),
            tokenizer_correct_answer.convert_tokens_to_ids('."'),
            tokenizer_correct_answer.convert_tokens_to_ids('\u0120"'),
        ],
    }
)


def generate_correct_answers_parallel(rank, *args):
    global logger
    logger = logging.getLogger(f"{__name__} (GPU {rank})")
    logger.info(f"Using GPU {rank} / {torch.cuda.device_count()}, args: {args}")

    squad = datasets.load_dataset("rajpurkar/squad", split="train")

    # As we're using multiple GPUs, we need to only take the part of the dataset that corresponds to the current GPU
    squad = datasets.distributed.split_dataset_by_node(
        squad, rank=rank, world_size=torch.cuda.device_count()
    )  # .take(10)

    logger.info(f"Dataset length: {len(squad)}")

    correct_answer_prompt_dataset = squad.map(
        lambda example: generate_correct_prompt(
            example["question"],
            example["answers"]["text"],
        ),
        num_proc=16,
    )

    # Create a dataloader
    correct_answer_prompt_tokenized_dataset = correct_answer_prompt_dataset.map(
        lambda example: tokenize_chat_correct_answer(example["correct_chat_template"]),
        num_proc=16,
        batched=False,
        # remove_columns=correct_answer_prompt_dataset.column_names,
    )

    correct_answer_prompt_tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask"]
    )

    data_collator_correct_answer = transformers.DataCollatorWithPadding(
        tokenizer=tokenizer_correct_answer,
        padding="max_length",
        return_tensors="pt",
        max_length=512,
    )

    # Drop all the examples that have length > 512
    correct_answer_prompt_tokenized_dataset = (
        correct_answer_prompt_tokenized_dataset.filter(
            lambda example: example["input_ids"].size(0) <= 512
        )
    )

    # logger.info the CUDA memory usage here
    logger.info(
        f"CUDA memory allocated before loading the correct answer generation model: {torch.cuda.memory_allocated(rank)}"
    )

    model_correct: transformers.LlamaForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_name_correct_answer
    )
    model_correct = model_correct.to(f"cuda:{rank}")
    model_correct.eval()

    # logger.info how much CUDA memory is available
    logger.info(
        f"CUDA memory allocated after loading the correct answer generation model: {torch.cuda.memory_allocated(rank)}"
    )

    # Generate the correct answer
    def generate_correct_answer(batch):
        # Collate the input_ids and attention_mask
        inputs_collated = data_collator_correct_answer(
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
        )
        input_ids = inputs_collated["input_ids"].to(model_correct.device)
        attention_mask = inputs_collated["attention_mask"].to(model_correct.device)
        logger.debug(f"Size of input_ids: {input_ids.size()}")
        outputs = model_correct.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=correct_answer_generation_config,
        ).to("cpu")

        # This has shape (batch_size * num_return_sequences, sequence_length)
        # First, we need to decode all the sequences
        # Only take the generated text
        generated_outputs = outputs[..., input_ids.size(1) :]
        # Remove the last token
        generated_outputs = generated_outputs[..., :-1]

        decoded = tokenizer_correct_answer.batch_decode(
            generated_outputs, skip_special_tokens=True
        )

        # The return dict will have:
        # {
        #     original_key_0: batch[original_key_0],
        #     correct_answer_generated: decoded[0],
        #     original_key_0: batch[original_key_0],
        #     correct_answer_generated: decoded[1],
        #     ...
        # }
        # In other words, we will augment the number of examples by the number of return sequences

        return_dict = {
            key: batch[key] * correct_answer_generation_config.num_return_sequences
            for key in (
                batch.keys() - ["input_ids", "attention_mask", "correct_chat_template"]
            )
        }

        # Now, add the generated text, but we need to take it as [0, num_return_sequences, 2 * num_return_sequences, ...] instead of [0, 1, 2, ...]
        return_dict["correct_answer_generated"] = []
        for num_return_sequence in range(
            correct_answer_generation_config.num_return_sequences
        ):
            for i in range(
                0, len(decoded), correct_answer_generation_config.num_return_sequences
            ):
                return_dict[f"correct_answer_generated"].append(
                    decoded[i + num_return_sequence]
                )

        return return_dict

    correct_answer_generated_dataset = correct_answer_prompt_tokenized_dataset.map(
        generate_correct_answer,
        batched=True,
        batch_size=32,
        remove_columns=["input_ids", "attention_mask", "correct_chat_template"],
    )

    logger.info(f"Dataset structure: {correct_answer_generated_dataset}")

    # Unload the model
    # logger.info how much CUDA memory is available
    logger.info(
        f"CUDA memory allocated before removing the correct answer generation model: {torch.cuda.memory_allocated(rank)}"
    )
    del model_correct
    gc.collect()
    torch.cuda.empty_cache()

    # logger.info how much CUDA memory is available
    logger.info(
        f"CUDA memory allocated after removing the correct answer generation model: {torch.cuda.memory_allocated(rank)} (it should be 0)"
    )

    # Remove the examples that are identical
    # Convert to Pandas and remove duplicates
    pandas_ds = correct_answer_generated_dataset.to_pandas()
    pandas_ds = pandas_ds.drop_duplicates("correct_answer_generated")
    correct_answer_generated_dataset = datasets.Dataset.from_pandas(
        pandas_ds, preserve_index=False
    )

    # Store the dataset
    correct_answer_generated_dataset.save_to_disk(
        f"correct_answer_generated_dataset_{rank}"
    )


def generate_hallucinated_answers_parallel(rank, *args):
    global logger
    logger = logging.getLogger(f"{__name__} (GPU {rank})")

    try:
        correct_answer_generated_dataset = datasets.load_from_disk(
            f"correct_answer_generated_dataset_{rank}"
        )  # .take(500)
    except FileNotFoundError:
        logger.info(
            f"Correct answer dataset for GPU {rank} not found. Please run the correct answer generation script first."
        )
        return

    logger.info(f"Dataset length: {len(correct_answer_generated_dataset)}")

    # Now, we will generate the hallucinated answers
    hallucination_prompt_dataset = correct_answer_generated_dataset.map(
        lambda example: generate_hallucination_prompt(
            example["question"],
            example["answers"]["text"],
            example["correct_answer_generated"],
        ),
        num_proc=16,
    )

    # Create a dataloader
    hallucination_prompt_tokenized_dataset = hallucination_prompt_dataset.map(
        lambda example: tokenize_chat_hallucination(
            example["hallucination_chat_template"]
        ),
        num_proc=16,
        batched=False,
        # remove_columns=correct_answer_prompt_dataset.column_names,
    )

    hallucination_prompt_tokenized_dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask"]
    )

    data_collator_hallucination = transformers.DataCollatorWithPadding(
        tokenizer=tokenizer_hallucination,
        padding="max_length",
        return_tensors="pt",
        max_length=768,
    )

    # Drop all the examples that have length > 800
    hallucination_prompt_tokenized_dataset = (
        hallucination_prompt_tokenized_dataset.filter(
            lambda example: example["input_ids"].size(0) <= 768
        )
    )

    # logger.info the CUDA memory usage here
    logger.info(
        f"CUDA memory allocated before loading the hallucination model: {torch.cuda.memory_allocated(rank)}"
    )

    model_hallucination: transformers.FalconForCausalLM = (
        AutoModelForCausalLM.from_pretrained(model_name_hallucination)
    )
    model_hallucination = model_hallucination.to(f"cuda:{rank}")
    model_hallucination.eval()

    # logger.info how much CUDA memory is available
    logger.info(
        f"CUDA memory allocated after loading the hallucination model: {torch.cuda.memory_allocated(rank)}"
    )

    # Generate the hallucinated answer
    def generate_hallucinated_answer(batch):
        inputs_collated = data_collator_hallucination(
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            }
        )
        input_ids = inputs_collated["input_ids"].to(model_hallucination.device)
        attention_mask = inputs_collated["attention_mask"].to(
            model_hallucination.device
        )
        logger.info(f"Size of input_ids: {input_ids.size()}")
        outputs = model_hallucination.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=hallucination_generation_config,
        ).to("cpu")
        # This has shape (batch_size * num_return_sequences, sequence_length)
        # First, we need to decode all the sequences
        # Only take the generated text
        generated_outputs = outputs[..., inputs_collated["input_ids"].size(1) :]
        # Remove the last token
        generated_outputs = generated_outputs[..., :-1]

        decoded = tokenizer_hallucination.batch_decode(
            generated_outputs, skip_special_tokens=True
        )

        # The return dict will have:
        # {
        #     original_key_0: batch[original_key_0],
        #     correct_answer_generated: decoded[0],
        #     original_key_0: batch[original_key_0],
        #     correct_answer_generated: decoded[1],
        #     ...
        # }
        # In other words, we will augment the number of examples by the number of return sequences

        return_dict = {
            key: batch[key] * hallucination_generation_config.num_return_sequences
            for key in (
                batch.keys()
                - ["input_ids", "attention_mask", "hallucination_chat_template"]
            )
        }

        # Now, add the generated text, but we need to take it as [0, num_return_sequences, 2 * num_return_sequences, ...] instead of [0, 1, 2, ...]
        return_dict["hallucinated_answer_generated"] = []
        for num_return_sequence in range(
            hallucination_generation_config.num_return_sequences
        ):
            for i in range(
                0, len(decoded), hallucination_generation_config.num_return_sequences
            ):
                return_dict[f"hallucinated_answer_generated"].append(
                    decoded[i + num_return_sequence]
                )

        return return_dict

    hallucinated_answer_generated_dataset = hallucination_prompt_tokenized_dataset.map(
        generate_hallucinated_answer,
        batched=True,
        batch_size=20,
        remove_columns=["input_ids", "attention_mask", "hallucination_chat_template"],
    )

    # Unload the model
    # logger.info how much CUDA memory is available
    logger.info(
        f"CUDA memory allocated before removing the hallucination model: {torch.cuda.memory_allocated(rank)}"
    )
    del model_hallucination
    gc.collect()
    torch.cuda.empty_cache()

    # logger.info how much CUDA memory is available
    logger.info(
        f"CUDA memory allocated after removing the hallucination model: {torch.cuda.memory_allocated(rank)} (it should be 0)"
    )

    # Remove the examples that are identical
    # Convert to Pandas and remove duplicates
    pandas_ds = hallucinated_answer_generated_dataset.to_pandas()
    pandas_ds = pandas_ds.drop_duplicates(
        ["correct_answer_generated", "hallucinated_answer_generated"]
    )
    hallucinated_answer_generated_dataset = datasets.Dataset.from_pandas(
        pandas_ds, preserve_index=False
    )

    # Store the dataset
    hallucinated_answer_generated_dataset.save_to_disk(
        f"hallucinated_answer_generated_dataset_{rank}"
    )


def remove_generation_artifacts(example):
    """
    Sometimes, there's some examples where the answer ends with ']', '}' or '",\s?'.
    Another example is:
    A reserving party to a treaty may include a statement that attempts to do what to its legal obligations or their effects ", "wrong": "A reserving party
    So we also want to catch the ",\s?" pattern and remove everything after it.
    We'll use regex to remove these artifacts.
    """
    # Keep only everything before the first occurrence of ']', '}', '",\s?\w*:'
    example["correct_answer_generated"] = re.sub(
        r"(\]|\}|\"\,\s?\w*:).*", "", example["correct_answer_generated"]
    )
    example["hallucinated_answer_generated"] = re.sub(
        r"(\]|\}|\"\,\s?\w*:).*", "", example["hallucinated_answer_generated"]
    )

    return example


def annotate_hallucination_spans(example):
    """
    Use a diff algorithm to find the differences between the correct answer and the hallucinated answer.
    """
    correct_answer = example["correct_answer_generated"]
    hallucinated_answer = example["hallucinated_answer_generated"]

    # Use the diff algorithm to find the differences
    try:
        diff = list(difflib.ndiff(correct_answer, hallucinated_answer))
    except Exception as e:
        logger.error(f"Error in diffing {example}: {e}")
        return {
            "hallucination_spans": [],
        }
    # We want tuples of (start character (inclusive), end character (not inclusive))
    hallucination_spans = []
    current_span = None
    for i, line in enumerate(diff):
        if line.startswith("+ "):
            if current_span is None:
                current_span = [i, i + 1]
            else:
                current_span[1] = i + 1
        else:
            if current_span is not None:
                hallucination_spans.append(tuple(current_span))
                current_span = None

    return {
        "hallucination_spans": hallucination_spans,
    }


if __name__ == "__main__":
    # Set spawn method
    multiprocessing.set_start_method("spawn")

    args = argparse.ArgumentParser()
    args.add_argument("--generate_correct_answers", action="store_true")
    args.add_argument("--generate_hallucinated_answers", action="store_true")
    args.add_argument("--merge_hallucinated_answers", action="store_true")
    args.add_argument("--generate_hallucination_annotations", action="store_true")
    args.add_argument("--remove_generation_artifacts", action="store_true")
    args.add_argument("--use_specific_cuda_gpu", type=int, default=None)

    args = args.parse_args()

    num_gpus = torch.cuda.device_count()

    # Fork this process into as many processes as there are GPUs
    # This is necessary because we will be loading the model in each process
    if args.generate_correct_answers:
        if args.use_specific_cuda_gpu is not None:
            generate_correct_answers_parallel(args.use_specific_cuda_gpu)
        else:
            torch_mp.spawn(
                generate_correct_answers_parallel,
                nprocs=num_gpus,
                join=True,
            )

    # Separate this in two steps because for some reason there is a bit of memory leak when we run both in the same process. This is a workaround.
    if args.generate_hallucinated_answers:
        if args.use_specific_cuda_gpu is not None:
            generate_hallucinated_answers_parallel(args.use_specific_cuda_gpu)
        else:
            torch_mp.spawn(
                generate_hallucinated_answers_parallel,
                nprocs=num_gpus,
                join=True,
            )

    # Merge the datasets
    if args.merge_hallucinated_answers:
        logger.info("Merging the datasets of the hallucinated answers")
        datasets_to_concatenate = []
        for i in range(num_gpus):
            datasets_to_concatenate.append(
                datasets.load_from_disk(f"hallucinated_answer_generated_dataset_{i}")
            )

        concatenated_datasets = datasets.concatenate_datasets(datasets_to_concatenate)

        # Split the dataset into training, validation, and test (80-10-10)
        concatenated_datasets = concatenated_datasets.train_test_split(
            test_size=0.2, seed=42
        )
        concatenated_datasets_test_val = concatenated_datasets["test"].train_test_split(
            test_size=0.5, seed=42
        )

        # Add them to the same dataset with different splits
        concatenated_datasets = datasets.DatasetDict(
            {
                "train": concatenated_datasets["train"],
                "validation": concatenated_datasets_test_val["train"],
                "test": concatenated_datasets_test_val["test"],
            }
        )

        concatenated_datasets.save_to_disk("hallucinated_answer_generated_dataset")

        # Push to the hub
        concatenated_datasets.push_to_hub(
            "shroom-semeval25/hallucinated_answer_generated_dataset",
            private=True,
        )

    # Remove the generation artifacts
    if args.remove_generation_artifacts:
        hallucinated_answer_generated_dataset = datasets.load_dataset(
            "shroom-semeval25/hallucinated_answer_generated_dataset"
        )

        hallucinated_answer_generated_dataset = (
            hallucinated_answer_generated_dataset.map(
                remove_generation_artifacts,
                num_proc=32,
            )
        )

        # Push to the hub
        hallucinated_answer_generated_dataset.push_to_hub(
            "shroom-semeval25/hallucinated_answer_generated_dataset_cleaned",
            private=True,
        )

    # Generate the annotations
    if args.generate_hallucination_annotations:
        raise NotImplementedError(
            "This part is not implemented yet. The annotations are generated at training time."
        )

        hallucinated_answer_generated_dataset = datasets.load_from_disk(
            "hallucinated_answer_generated_dataset_cleaned"
        )

        annotated_dataset = hallucinated_answer_generated_dataset.map(
            annotate_hallucination_spans,
            num_proc=32,
        )

        # Filter out the examples that have no hallucinated spans
        annotated_dataset = annotated_dataset.filter(
            lambda example: len(example["hallucination_spans"]) > 0,
            num_proc=32,
        )

        annotated_dataset.save_to_disk(
            "hallucinated_answer_generated_dataset_cleaned_annotated"
        )

        # Push to the hub
        annotated_dataset.push_to_hub(
            "shroom-semeval25/hallucinated_answer_generated_dataset_cleaned_annotated",
            private=True,
        )
