# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2025) B-
# ytedance Inc..
# *************************************************************************

# Adapted from https://github.com/NVlabs/describe-anything/blob/main/evaluation/eval_model_outputs.py

# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import base64
import io
import json
import os

import inflect
import numpy as np
import openai
from PIL import Image
from pycocotools.coco import COCO
from tqdm import tqdm

# Define Azure OpenAI details
model_name = "gpt-4o-2024-11-20"
max_tokens = 1000  # range: [1, 4095]

# Initialize the Azure client
client = openai.AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version="2024-03-01-preview",
)

prompt_eval = """Answer the multiple-choice question based on the text description of an object in this image. You need to follow these rules:
1. Do not output any reasoning. Do not perform correction. Please output exactly one answer from the choices for each question. Do not repeat the question.
2. There is no need for exact matching. Please choose the closest option based on the description.

The description is:
{pred_caption}

From the description above, please answer the following question with one of the choices:
{question_text_str}
"""

api_call_count = 0


def query(prompt, images, temperature, max_tokens):
    global api_call_count
    if api_call_count >= args.api_call_limit:
        raise Exception("API call limit reached")

    api_call_count += 1
    content = [
        {"type": "text", "text": "The image:\n"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{images[0]}"},
        },
        {"type": "text", "text": "\nThe mask of the image:\n"},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{images[1]}"},
        },
        {"type": "text", "text": f"\n{prompt}\n"},
    ]

    # Adjusted to use the Azure OpenAI client with the specified parameters
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    message = response.choices[0].message.content
    return message


def parse_pred(pred, choices, key):
    pred = pred.strip().lower()
    substr_indices = []
    for index, choice in enumerate(choices):
        choice = choice.strip().lower()
        prefix = "abcde"[index]
        if choice == pred or pred == f"{prefix}. {choice}" or pred == prefix:
            return index
        if choice in pred:
            substr_indices.append((index, pred.index(choice), len(choice)))

    if len(substr_indices) == 1:
        return substr_indices[0][0]

    choices_label = "abcde"
    if pred[0] in choices_label and pred[1] == ".":
        ret = choices_label.index(pred[0])
        return ret

    if substr_indices:
        if len(substr_indices) > 1:
            ret, ret_pos, _ = max(substr_indices, key=lambda x: x[1])
            max_items = [item for item in substr_indices if item[1] == ret_pos]
            if len(max_items) > 1:
                ret = max(max_items, key=lambda x: x[2])[0]
            return ret
        else:
            ret = substr_indices[0][0]
        return ret

    match_lengths = []
    for index, choice in enumerate(choices):
        choice = choice.strip().lower()
        if pred in choice:
            match_lengths.append((index, len(choice)))
    if match_lengths:
        if len(match_lengths) > 1:
            ret = max(match_lengths, key=lambda x: x[1])[0]
        else:
            ret = match_lengths[0][0]
        return ret

    if pred[0] in "abcde" and (len(pred.strip()) == 1 or pred[1] == "\n"):
        ret = "abcde".index(pred[0])
        return ret

    return None


def evaluate(
    question_dicts,
    pred_caption,
    temperature,
    max_tokens,
    images,
    *,
    response_override=None,
    key,
    verbose=False,
) -> dict:
    pred_answers = []
    prompt = []
    response = []
    for index, question_dict in enumerate(question_dicts):
        question_text_str = f"{question_dict['question']}\n"
        choices_text = ""
        for choice_index, (choice, score) in enumerate(question_dict["choices"]):
            choice_index = "ABCDE"[choice_index]
            choices_text += f"{choice_index}. {choice}\n"
        question_text_str += choices_text
        prompt_item = prompt_eval.format(
            pred_caption=pred_caption, question_text_str=question_text_str.strip()
        )

        if (
            response_override is None
            or len(response_override) < index
            or response_override[index] is None
        ):
            response_item = query(prompt_item, images, temperature, max_tokens)
        else:
            response_item = response_override[index]

        pred_answer = response_item.strip()
        pred_answers.append(pred_answer)
        prompt.append(prompt_item)
        response.append(response_item)

    pred_indices = [
        parse_pred(
            pred_answer, [choice for choice, score in question_dict["choices"]], key
        )
        for pred_answer, question_dict in zip(pred_answers, question_dicts)
    ]
    parsed_eval_results = [
        question_dict["choices"][pred_index][1] if pred_index is not None else 0
        for pred_index, question_dict in zip(pred_indices, question_dicts)
    ]

    parsed_eval_results_positives = []
    parsed_eval_results_negatives = []
    details_positives = []
    details_negatives = []
    details_recognition = []
    recognition_result = None
    for question_index, (parsed_eval_result, question_dict) in enumerate(
        zip(parsed_eval_results, question_dicts)
    ):
        if question_dict["type"] == "recognition":
            if parsed_eval_result == "correct":
                recognition_result = True
            elif parsed_eval_result == "incorrect":
                recognition_result = False
                print(
                    f"Recognition is incorrect for key {key}, setting score to at most 0 for all questions"
                )
            else:
                raise ValueError(f"Invalid recognition result: {parsed_eval_result}")
            details_recognition.append(
                {
                    **question_dict,
                    "pred_answer": pred_answers[question_index],
                    "pred_index": pred_indices[question_index],
                    "eval_result": parsed_eval_result,
                }
            )
        elif question_dict["type"] == "negative":
            if recognition_result is False:
                parsed_eval_result = min(0, parsed_eval_result)
            parsed_eval_results_negatives.append(parsed_eval_result)

            details_negatives.append(
                {
                    **question_dict,
                    "pred_answer": pred_answers[question_index],
                    "pred_index": pred_indices[question_index],
                    "eval_result": parsed_eval_result,
                }
            )
        elif question_dict["type"] == "positive":
            if recognition_result is False:
                parsed_eval_result = min(0, parsed_eval_result)
            parsed_eval_results_positives.append(parsed_eval_result)

            details_positives.append(
                {
                    **question_dict,
                    "pred_answer": pred_answers[question_index],
                    "pred_index": pred_indices[question_index],
                    "eval_result": parsed_eval_result,
                }
            )

    score_pos = sum(parsed_eval_results_positives) / len(parsed_eval_results_positives)
    score_neg = (
        sum(parsed_eval_results_negatives) / len(parsed_eval_results_negatives)
        if parsed_eval_results_negatives
        else None
    )
    score = (
        sum(parsed_eval_results_positives) + sum(parsed_eval_results_negatives)
    ) / (len(parsed_eval_results_positives) + len(parsed_eval_results_negatives))

    info = dict(
        details_positives=details_positives,
        details_negatives=details_negatives,
        details_recognition=details_recognition,
        prompt=prompt,
        response=response,
        score=score,
        score_pos=score_pos,
        score_neg=score_neg,
        recognition_result=recognition_result,
    )

    return info


def is_plural(string):
    if string == "bus":
        return False
    return p.singular_noun(string) is not False


def select_ann(img_id, area_min=None, area_max=None):
    cat_ids = coco.getCatIds()
    ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)

    if area_min is not None:
        ann_ids = [
            ann_id for ann_id in ann_ids if coco.anns[ann_id]["area"] >= area_min
        ]

    if area_max is not None:
        ann_ids = [
            ann_id for ann_id in ann_ids if coco.anns[ann_id]["area"] <= area_max
        ]

    return ann_ids


def mask_to_box(mask_np):
    mask_coords = np.argwhere(mask_np)
    y0, x0 = mask_coords.min(axis=0)
    y1, x1 = mask_coords.max(axis=0) + 1

    h = y1 - y0
    w = x1 - x0

    return x0, y0, w, h


def encode_pil_image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model outputs")
    parser.add_argument(
        "--pred", type=str, help="Path to the prediction JSON file", required=True
    )
    parser.add_argument(
        "--qa",
        type=str,
        help="Path to the reference QA file",
        default="evaluation/DLC-Bench/annotations/qa.json",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        help="Path to the class names JSON file",
        default="evaluation/DLC-Bench/annotations/class_names.json",
    )
    parser.add_argument(
        "--api-call-limit", type=int, default=1000, help="API call limit"
    )
    parser.add_argument(
        "--suffix", type=str, default="", help="Suffix for the evaluation file"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose mode")
    parser.add_argument(
        "--quiet", action="store_true", help="Enable quiet mode (result only)"
    )
    parser.add_argument("--csv", action="store_true", help="Output results as CSV only")
    parser.add_argument(
        "--data-root", type=str, default="evaluation/DLC-Bench/annotations"
    )

    args = parser.parse_args()

    eval_file = os.path.splitext(args.pred)[0] + f"_eval_gpt{args.suffix}.json"

    eval_results = {}

    if os.path.exists(eval_file):
        with open(eval_file) as f:
            eval_results = json.load(f)

    with open(args.pred) as f:
        data_pred = json.load(f)

    with open(args.qa) as f:
        data_qa = json.load(f)

    with open(args.class_names) as f:
        data_class_names = json.load(f)

    scores = {}
    scores_pos = {}
    scores_neg = {}

    keys = list(data_qa.keys())
    p = inflect.engine()

    annotations_file = os.path.join(args.data_root, "annotations.json")
    coco = COCO(annotations_file)

    with open(annotations_file, "r") as f:
        data = json.load(f)

    missing_key_count = 0
    for key in tqdm(keys, disable=args.quiet):
        key = str(key)
        for item in data["annotations"]:
            if int(item["id"]) == int(key):
                img_id = item["image_id"]

        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(args.data_root, "images", img_info["file_name"])
        img = Image.open(img_path)

        anns = coco.loadAnns([int(key)])
        mask_np = coco.annToMask(anns[0]).astype(bool)

        img_np = np.array(img)
        pil_mask = Image.fromarray((mask_np * 255).astype(np.uint8))

        assert (
            img_np.shape[:2] == mask_np.shape
        ), f"image shape mismatches with mask shape: {img_np.shape}, {mask_np.shape}"
        img_h, img_w = img_np.shape[:2]

        x0, y0, w, h = mask_to_box(mask_np)
        xc, yc = x0 + w / 2, y0 + h / 2

        # focal_crop: need to have at least min_box_w and min_box_h pixels, otherwise resizing to (384, 384) leads to artifacts that may be OOD
        w, h = max(w, 56), max(h, 56)
        x0, y0 = int(xc - w / 2), int(yc - h / 2)

        # focal crop
        cropped_img_np = img_np[
            max(y0 - h, 0) : min(y0 + 2 * h, img_h),
            max(x0 - w, 0) : min(x0 + 2 * w, img_w),
        ]
        cropped_mask_np = mask_np[
            max(y0 - h, 0) : min(y0 + 2 * h, img_h),
            max(x0 - w, 0) : min(x0 + 2 * w, img_w),
        ]

        cropped_pil_img = Image.fromarray(cropped_img_np)
        cropped_pil_mask = Image.fromarray((cropped_mask_np * 255).astype(np.uint8))

        base64_image = encode_pil_image_to_base64(img)
        base64_mask = encode_pil_image_to_base64(pil_mask)
        base64_cropped_image = encode_pil_image_to_base64(cropped_pil_img)
        base64_cropped_mask = encode_pil_image_to_base64(cropped_pil_mask)
        images = [base64_cropped_image, base64_cropped_mask]

        if key in eval_results:
            response_override = eval_results[key]["response"]
        else:
            response_override = None

        if key not in data_pred:
            if args.default_prediction is None:
                raise ValueError(f"Key {key} not found in prediction data")
            else:
                pred_value = args.default_prediction
                missing_key_count += 1
        else:
            pred_value = data_pred[key]

        class_name = data_class_names[key]
        recognition_question = f"The object in the image is {class_name}. Based on the image, is it likely that the object in the description is given class: {class_name} or object of a similar type?"
        recognition_question_dict = {
            "question": recognition_question,
            "choices": [("Yes", "correct"), ("No", "incorrect")],
            "type": "recognition",
        }

        question_dicts = [recognition_question_dict, *data_qa[key]]
        info = evaluate(
            question_dicts=question_dicts,
            pred_caption=pred_value,
            images=images,
            temperature=0.0,
            max_tokens=300,
            response_override=response_override,
            key=key,
        )
        score = info["score"]
        scores[key] = score
        scores_pos[key] = info["score_pos"]
        scores_neg[key] = info["score_neg"]
        eval_results[key] = {"pred": pred_value, **info}

    avg_score_pos = sum(scores_pos.values()) / len(scores_pos)
    avg_score_neg = sum(
        [item for item in scores_neg.values() if item is not None]
    ) / len(scores_neg)
    eval_results["avg_pos"] = avg_score_pos
    eval_results["avg_neg"] = avg_score_neg

    with open(eval_file, "w") as f:
        json.dump(eval_results, f, indent=4)

    print(f"Average Positive Score: {avg_score_pos:.3f}")
    print(f"Average Negative Score: {avg_score_neg:.3f}")
    print(
        f"Summary (Pos\tNeg\tAvg(Pos, Neg)):\t{avg_score_pos:.3f},\t{avg_score_neg:.3f},\t{(avg_score_pos + avg_score_neg) / 2:.3f}"
    )
    print(f"QA Scores: {scores}")
    print(f"Evaluation data saved to {eval_file}")
