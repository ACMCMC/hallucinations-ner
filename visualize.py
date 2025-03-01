import gradio as gr
import datasets
import difflib
import transformers
import torch
import logging
from utils import predict_logits_q_a_model, trim_to_answer_only, get_hard_labels


tokenizer = transformers.AutoTokenizer.from_pretrained("google/flan-t5-base")

dataset = (
    datasets.load_dataset(
        "shroom-semeval25/hallucinated_answer_generated_dataset",
        split="test",
    )
    .take(10000)
    .to_pandas()
    .sort_values("question")
)

# Show columns in this order: question, correct_answer_generated, hallucinated_answer_generated, everything else
dataset = dataset[
    ["question", "correct_answer_generated", "hallucinated_answer_generated"]
    + [
        col
        for col in dataset.columns
        if col
        not in ["question", "correct_answer_generated", "hallucinated_answer_generated"]
    ]
]


def show_hallucinations(element):
    original_text = element["correct_answer_generated"]
    hallucinated_text = element["hallucinated_answer_generated"]
    # tokenize both texts
    original_tokens = tokenizer(
        original_text, return_offsets_mapping=True, add_special_tokens=False
    )
    hallucinated_tokens = tokenizer(
        hallucinated_text, return_offsets_mapping=True, add_special_tokens=False
    )
    # Find the tokens that are different. We have two lists of numbers, we need to find the differences (mind the order)
    diff = difflib.SequenceMatcher(
        None,
        original_tokens["input_ids"],
        hallucinated_tokens["input_ids"],
    ).get_opcodes()
    entities = []
    # Follows this structure:
    # {
    #     "entity": "+" or "-",
    #     "start": 0,
    #     "end": 0,
    # }
    for tag, i1, i2, j1, j2 in diff:
        try:
            if tag == "equal":
                continue
            # Anything that is not equal is a hallucination

            start_char = hallucinated_tokens["offset_mapping"][j1][0]
            end_char = hallucinated_tokens["offset_mapping"][j2 - 1][1] + 1
            entity = {
                "entity": "hal",
                "start": start_char,
                "end": end_char,
            }
            # entity_2 = {
            #     "entity": "-",
            #     "start": start,
            #     "end": end,
            # }
            entities.append(entity)
            # entities.append(entity_2)
        except IndexError as e:
            gr.Error(f"There was an error in the tokenization process: {e}")

    return [
        {
            "calculated_diffs": diff,
            "tokenized_original": original_tokens,
            "tokenized_hallucinated": hallucinated_tokens,
            **element.to_dict(),
        },
        element["correct_answer_generated"],
        {
            "text": hallucinated_text,
            "entities": entities,
        },
    ]


roberta_base_predictor = transformers.AutoModelForTokenClassification.from_pretrained(
    "shroom-semeval25/cogumelo-hallucinations-detector-roberta-base"
)
roberta_base_tokenizer = transformers.AutoTokenizer.from_pretrained(
    "shroom-semeval25/cogumelo-hallucinations-detector-roberta-base"
)
roberta_large_qa_predictor = (
    transformers.AutoModelForTokenClassification.from_pretrained(
        "shroom-semeval25/cogumelo-hallucinations-detector-roberta-large-qa-15000"
    )
)
flan_t5_qa_predictor = transformers.AutoModelForTokenClassification.from_pretrained(
    "shroom-semeval25/cogumelo-hallucinations-detector-flan-t5-xl-qa-v3"
)
flan_t5_qa_tokenizer = transformers.AutoTokenizer.from_pretrained(
    "shroom-semeval25/cogumelo-hallucinations-detector-flan-t5-xl-qa-v3"
)


def update_selection(evt: gr.SelectData):
    selected_row = evt.index[0]
    element = dataset.iloc[selected_row]
    question = element["question"]
    hallucinated_answer = element["hallucinated_answer_generated"]
    json_example, original_text, highlighted_text = show_hallucinations(element)
    return (
        json_example,
        original_text,
        highlighted_text,
        *get_hallucinations(hallucinated_answer, question),
    )


def get_hallucinations(hallucinated_answer: str, question: str):
    try:
        hallucinated_tokens = roberta_base_tokenizer(
            text=hallucinated_answer,
            return_offsets_mapping=True,
            add_special_tokens=True,
            return_tensors="pt",
            return_special_tokens_mask=True,
        )
        with torch.no_grad():
            outputs_roberta_base = roberta_base_predictor(
                input_ids=hallucinated_tokens.input_ids,
                attention_mask=hallucinated_tokens.attention_mask,
            )
            # Take only the outputs that are NOT special tokens and where the attention mask is 1
            logits_roberta_base = outputs_roberta_base.logits[
                ...,
                (hallucinated_tokens.special_tokens_mask == 0)
                & (hallucinated_tokens.attention_mask == 1),
                :,
            ]
        hard_labels_predicted_roberta_base = get_hard_labels(
            logits=logits_roberta_base,
            # Discard the first token, which is the BOS token
            offsets=hallucinated_tokens["offset_mapping"][0][1:],
        )
        highlighted_text_predicted_roberta_base = [
            {
                "entity": "hal",
                **x,
            }
            for x in hard_labels_predicted_roberta_base
        ]
        if question:
            logits_roberta_large_qa, offsets_roberta_large_qa = (
                predict_logits_q_a_model(
                    model=roberta_large_qa_predictor,
                    tokenizer=roberta_base_tokenizer,
                    question=question,
                    answer=hallucinated_answer,
                )
            )

            logits_flan_t5_qa, offsets_flan_t5_qa = predict_logits_q_a_model(
                model=flan_t5_qa_predictor,
                tokenizer=flan_t5_qa_tokenizer,
                question=question,
                answer=hallucinated_answer,
            )

            hard_labels_predicted_roberta_large_qa = get_hard_labels(
                logits=logits_roberta_large_qa,
                offsets=offsets_roberta_large_qa,
            )
            hard_labels_predicted_flan_t5_qa = get_hard_labels(
                logits=logits_flan_t5_qa,
                offsets=offsets_flan_t5_qa,
            )
            highlighted_text_predicted_roberta_large_qa = [
                {
                    "entity": "hal",
                    **x,
                }
                for x in hard_labels_predicted_roberta_large_qa
            ]
            highlighted_text_predicted_flan_t5_qa = [
                {
                    "entity": "hal",
                    **x,
                }
                for x in hard_labels_predicted_flan_t5_qa
            ]
        else:
            highlighted_text_predicted_roberta_large_qa = {"text": "", "entities": []}
            highlighted_text_predicted_flan_t5_qa = {"text": "", "entities": []}
    except Exception as e:
        logging.exception(f"An error occurred: {e}")
        gr.Error(f"An error occurred: {e}")
        highlighted_text_predicted_roberta_base = {"text": "", "entities": []}
        highlighted_text_predicted_roberta_large_qa = {"text": "", "entities": []}
        highlighted_text_predicted_flan_t5_qa = {"text": "", "entities": []}
    return (
        highlighted_text_predicted_roberta_base,
        highlighted_text_predicted_roberta_large_qa,
        highlighted_text_predicted_flan_t5_qa,
    )


def predict_hallucinations_manual_input(text: str, question: str = ""):
    return get_hallucinations(text, question)


with gr.Blocks(title="Hallucinations Explorer") as demo:
    # A selectable dataframe with the dataset
    # print(dataset)
    gr.Markdown(
        """# COGUMELO
                
_SHROOM '25: Detection of Hallucinated Content_"""
    )

    with gr.Accordion(label="Manual Input", open=True) as manual_input:
        model_question_input = gr.Textbox(
            value="",
            label="Question (only for RoBERTa Large QA and Flan T5 QA)",
            placeholder="Type the question here",
            type="text",
        )

        # A textbox where the user can input any text to try the model
        model_manual_input = gr.Textbox(
            value="",
            label="Try your own text",
            placeholder="Type your own text here",
            type="text",
        )

        manual_input_highlighted_text_roberta_base = gr.HighlightedText(
            label="Predicted Hallucinations (RoBERTa Base)",
            color_map={"+": "red", "-": "blue", "hal": "red"},
            combine_adjacent=True,
        )

        manual_input_highlighted_text_roberta_large_qa = gr.HighlightedText(
            label="Predicted Hallucinations (RoBERTa Large QA)",
            color_map={"+": "red", "-": "blue", "hal": "red"},
            combine_adjacent=True,
        )

        manual_input_highlighted_text_flan_t5_qa = gr.HighlightedText(
            label="Predicted Hallucinations (Flan T5 QA)",
            color_map={"+": "red", "-": "blue", "hal": "red"},
            combine_adjacent=True,
        )

        model_manual_input.change(
            predict_hallucinations_manual_input,
            inputs=[model_manual_input, model_question_input],
            outputs=[
                manual_input_highlighted_text_roberta_base,
                manual_input_highlighted_text_roberta_large_qa,
                manual_input_highlighted_text_flan_t5_qa,
            ],
        )

        # model_question_input.change(
        #     predict_hallucinations_manual_input_roberta_qa_large,
        #     inputs=[model_manual_input, model_question_input],
        #     outputs=[
        #         manual_input_highlighted_text_roberta_large_qa,
        #     ],
        # )

    gr.Markdown(
        """# Dataset
⚠️ These rows are part of the **test set** of the dataset, not the entire dataset (the model has therefore not seen them)
"""
    )
    df = gr.Dataframe(dataset)

    original_text = gr.Textbox(label="Original Text", interactive=False)
    highlighted_text = gr.HighlightedText(
        label="Real Hallucinations (ground truth)",
        color_map={"+": "red", "-": "blue", "hal": "red"},
        combine_adjacent=True,
    )
    highlighted_text_predicted_roberta_base = gr.HighlightedText(
        label="Predicted Hallucinations (RoBERTa Base)",
        color_map={"+": "red", "-": "blue", "hal": "red"},
        combine_adjacent=True,
    )
    highlighted_text_predicted_roberta_large_qa = gr.HighlightedText(
        label="Predicted Hallucinations (RoBERTa Large QA)",
        color_map={"+": "red", "-": "blue", "hal": "red"},
        combine_adjacent=True,
    )
    highlighted_text_predicted_flan_t5_qa = gr.HighlightedText(
        label="Predicted Hallucinations (Flan T5 QA)",
        color_map={"+": "red", "-": "blue", "hal": "red"},
        combine_adjacent=True,
    )
    json_example = gr.JSON()

    df.select(
        update_selection,
        inputs=[],
        outputs=[
            json_example,
            original_text,
            highlighted_text,
            highlighted_text_predicted_roberta_base,
            highlighted_text_predicted_roberta_large_qa,
            highlighted_text_predicted_flan_t5_qa,
        ],
    )


demo.launch(show_error=True)
