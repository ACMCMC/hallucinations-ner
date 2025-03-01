import torch
import difflib
import logging
import transformers
import datasets

logger = logging.getLogger(__name__)


CLASS_LABELS = datasets.ClassLabel(
    names=[
        "O",
        "B-HALLUCINATION",
        "I-HALLUCINATION",
    ]
)


def compute_diff(
    correct_tokens_example: torch.Tensor,
    hallucinated_tokens_example: torch.Tensor,
    CLASS_LABELS: datasets.ClassLabel = CLASS_LABELS,
):
    """
    Computes the differences between two tokenized texts and returns the entities that are hallucinated.
    """
    diff = difflib.SequenceMatcher(
        None,
        correct_tokens_example.tolist(),
        hallucinated_tokens_example.tolist(),
    ).get_opcodes()
    # By default, a vector of all CLASS_LABELS["OUTSIDE"]
    label = torch.full(
        (hallucinated_tokens_example.size(0),),
        CLASS_LABELS.str2int("O"),
        dtype=torch.long,
    )
    for tag, i1, i2, j1, j2 in diff:
        # Here, we'll set the labels for the hallucinated tokens
        try:
            if tag == "equal":
                continue
            if tag == "delete":
                # We can't tag anything as hallucinated if it was deleted because there's no specific token to tag
                continue
            # Anything that is not equal is a hallucination

            # Set the label for the hallucinated tokens
            # The first token is the beginning of the hallucination
            label[j1] = CLASS_LABELS.str2int("B-HALLUCINATION")
            # The rest of the tokens are inside the hallucination
            label[j1 + 1 : j2] = CLASS_LABELS.str2int("I-HALLUCINATION")
        except IndexError as e:
            logger.exception(
                "There was an error in the generation of hallucination labels"
            )

    return label


def get_hallucinations_pair(
    tokenized_pair_correct: torch.Tensor,
    attention_mask_pair_correct: torch.Tensor,
    token_type_ids_pair_correct: torch.Tensor,
    special_tokens_mask_pair_correct: torch.Tensor,
    tokenized_pair_hallucinated: torch.Tensor,
    attention_mask_pair_hallucinated: torch.Tensor,
    token_type_ids_pair_hallucinated: torch.Tensor,
    special_tokens_mask_pair_hallucinated: torch.Tensor,
    CLASS_LABELS: datasets.ClassLabel = CLASS_LABELS,
):
    """
    Like get_hallucinations, but for a pair of tokenized texts (question and answer).
    We will use the `token_type_ids` to separate the two texts and run the diff on the second one (the answer).
    """
    # For each element in the batch, separate the two texts and run the diff
    # Assert that the tokens have a batch dimension
    assert tokenized_pair_correct.dim() == 2
    assert attention_mask_pair_correct.dim() == 2
    assert token_type_ids_pair_correct.dim() == 2
    assert tokenized_pair_hallucinated.dim() == 2
    assert attention_mask_pair_hallucinated.dim() == 2
    assert token_type_ids_pair_hallucinated.dim() == 2
    labels = []

    for i in range(tokenized_pair_correct.size(0)):
        # Separate using the token type IDs and exclude whatever is not in the attention mask and is not a special token
        correct_tokens_element = tokenized_pair_correct[i][
            (token_type_ids_pair_correct[i] == 1)
            & (attention_mask_pair_correct[i] == 1)
            & (special_tokens_mask_pair_correct[i] == 0)
        ]
        hallucinated_tokens_element = tokenized_pair_hallucinated[i][
            (token_type_ids_pair_hallucinated[i] == 1)
            & (attention_mask_pair_hallucinated[i] == 1)
            & (special_tokens_mask_pair_hallucinated[i] == 0)
        ]

        # Compute the diff
        label = compute_diff(
            correct_tokens_example=correct_tokens_element,
            hallucinated_tokens_example=hallucinated_tokens_element,
            CLASS_LABELS=CLASS_LABELS,
        )

        # Turn the label into a tensor with the same shape as the padded version (i.e. we need to add the padding tokens)
        PADDING_TOKEN = -100
        padded_label = torch.full(
            (tokenized_pair_hallucinated[i].size(0),),
            PADDING_TOKEN,
            dtype=torch.long,
        )
        padded_label[
            (token_type_ids_pair_hallucinated[i] == 1)
            & (attention_mask_pair_hallucinated[i] == 1)
            & (special_tokens_mask_pair_hallucinated[i] == 0)
        ] = label

        labels.append(padded_label)

    return torch.stack(labels)


def get_hallucinations_correct_and_hallucinated(
    correct_tokens: torch.Tensor,
    correct_tokens_attention_mask: torch.Tensor,
    hallucinated_tokens: torch.Tensor,
    hallucinated_tokens_attention_mask: torch.Tensor,
    CLASS_LABELS: datasets.ClassLabel = CLASS_LABELS,
):
    """
    Computes the differences between two tokenized texts and returns the entities that are hallucinated.
    """
    # Find the tokens that are different. We have two lists of numbers, we need to find the differences (mind the order)
    # Assert that the tokens have a batch dimension
    assert correct_tokens.dim() == 2
    assert hallucinated_tokens.dim() == 2
    labels = []

    # For each element in the batch
    for i in range(correct_tokens.size(0)):
        # Exclude whatever is not in the attention mask
        correct_tokens_element = correct_tokens[i][
            correct_tokens_attention_mask[i] == 1
        ]
        hallucinated_tokens_element = hallucinated_tokens[i][
            hallucinated_tokens_attention_mask[i] == 1
        ]

        # Compute the diff
        label = compute_diff(
            correct_tokens_example=correct_tokens_element,
            hallucinated_tokens_example=hallucinated_tokens_element,
            CLASS_LABELS=CLASS_LABELS,
        )

        # Turn the label into a tensor with the same shape as the padded version (i.e. we need to add the padding tokens)
        PADDING_TOKEN = -100
        padded_label = torch.full(
            (hallucinated_tokens[i].size(0),),
            PADDING_TOKEN,
            dtype=torch.long,
        )
        padded_label[hallucinated_tokens_attention_mask[i] == 1] = label

        labels.append(padded_label)

    return torch.stack(labels)


def trim_to_answer_only(
    logits: torch.Tensor, special_tokens_mask: torch.Tensor, offsets: torch.Tensor
):
    token_type_ids = torch.zeros_like(special_tokens_mask)
    # Do a cumsum on the special tokens mask
    cumsum_special_tokens_mask = special_tokens_mask.cumsum(dim=-1)
    # Restrict it to the non-special tokens
    cumsum_special_tokens_mask = cumsum_special_tokens_mask[
        special_tokens_mask == 0
    ]  # Now this has a different shape, we don't know what exactly it is, but we'll use it consistently so it's fine
    # In case that the first token is a special token, we'd have cumsums on non-special tokens that are 1 and 2; otherwise, we'd have 0 and 1. To make sure that we always have 0 and 1, we'll subtract the minimum
    cumsum_special_tokens_mask -= cumsum_special_tokens_mask.min()
    token_type_ids[special_tokens_mask == 0] = cumsum_special_tokens_mask
    return logits[token_type_ids != 0], offsets[token_type_ids != 0], token_type_ids


def predict_logits_q_a_model(model, tokenizer, question, answer):
    """
    Predicts the labels for a question and an answer.
    """

    q_a_tokens = tokenizer(
        # We have to batch into a single-example batch, because otherwise the tokenizer will interpret that the second element of the pair is example #2 of the batch (while actually it is the second part of the pair of example #1)
        text=[(question, answer)],
        return_offsets_mapping=True,
        add_special_tokens=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    with torch.no_grad():
        logits = model(
            input_ids=q_a_tokens.input_ids,
            attention_mask=q_a_tokens.attention_mask,
        ).logits
        # Take only the outputs after the first special token and where the attention mask is 1 and the special tokens mask is 0
        logits, offsets, token_type_ids = trim_to_answer_only(
            logits=logits,
            special_tokens_mask=q_a_tokens.special_tokens_mask,
            offsets=q_a_tokens["offset_mapping"],
        )

    return logits, offsets


def get_hard_labels(logits, offsets):
    # Get the highest value for each token
    predictions = logits.argmax(dim=-1).squeeze(0).tolist()
    list_of_hard_labels = []
    current_hallucination = None
    for i, prediction in enumerate(predictions):
        if prediction == 0:
            if current_hallucination is not None:
                list_of_hard_labels.append(current_hallucination)
                current_hallucination = None
            continue
        if prediction == 1:
            if current_hallucination is not None:
                list_of_hard_labels.append(current_hallucination)
            current_hallucination = {
                "start": offsets[i][0],
                "end": offsets[i][1] + 1,
            }
        if prediction == 2:
            if current_hallucination is None:
                current_hallucination = {
                    "start": offsets[i][0],
                    "end": offsets[i][1] + 1,
                }
            else:
                current_hallucination["end"] = offsets[i][1] + 1
    if current_hallucination is not None:
        list_of_hard_labels.append(current_hallucination)
    # Convert the Tensor items to Python numbers
    list_of_hard_labels = [
        {
            "start": span["start"].item(),
            "end": span["end"].item(),
        }
        for span in list_of_hard_labels
    ]
    return list_of_hard_labels
