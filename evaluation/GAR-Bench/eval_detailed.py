# --------------------------------------------------------
# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License")
# Grasp Any Region Project
# Written by Haochen Wang and Yuhao Wang
# --------------------------------------------------------

import argparse
import base64
import io
import json
import os
import re

import numpy as np
import openai
from PIL import Image
from pycocotools import mask as mask_utils
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

prompt_ann = """
You are a language model expert. Your task is to evaluate the following model output based on the provided images, and subject, object, and relationship.

- subject_name: {subject_name}
- object_name: {object_name}
- predicate_name: {predicate_name}
- model_output: {model_output}

Task:
1. Check if the model output describes the {subject_name}. 
2. Check if the model output conveys the relationship between {subject_name} and {object_name} related to {predicate_name}.

Note:
- The first task only requires checking if {subject_name} is mentioned in the model output.
- The second task asks if the output conveys a relationship related to {predicate_name} between {subject_name} and {object_name}, even if different words or phrases are used.
- If both tasks are successfully completed, return "True" Otherwise, return "False"
- Do not output any reasoning. Do not perform correction. Please output only just one "True" or "False".

"""


def process_questions(outputs):

    pattern = r"^```json\s*|\s*```$"
    try:
        cleaned_str = re.sub(pattern, "", outputs, flags=re.MULTILINE)
        questions_data = json.loads(cleaned_str)
    except:
        print("Error in parsing JSON")
        return []
    return questions_data


def encode_pil_image_to_base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def mask_to_box(mask_np):
    mask_coords = np.argwhere(mask_np)
    y0, x0 = mask_coords.min(axis=0)
    y1, x1 = mask_coords.max(axis=0) + 1

    h = y1 - y0
    w = x1 - x0

    return x0, y0, w, h


def query(messages):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model outputs")
    parser.add_argument("--pred", type=str, help="Path to the model")
    parser.add_argument("--min_box_w", type=int, help="Minimum width", default=56)
    parser.add_argument("--min_box_h", type=int, help="Minimum height", default=56)
    parser.add_argument(
        "--image_folder", type=str, default="evaluation/GAR-Bench/annotations"
    )
    args = parser.parse_args()

    with open(args.pred, "r") as f:
        data = json.load(f)

    output_json = []
    total = 0
    true = 0

    for item in tqdm(data):
        total = total + 1
        model_output = item["model_output"]

        subject_name = item["subject_name"]
        object_name = item["object_name"]
        predicate_name = item["predicate_name"]
        model_output = item["model_output"]
        prompt = prompt_ann.format(
            subject_name=subject_name,
            object_name=object_name,
            predicate_name=predicate_name,
            model_output=model_output,
        )

        img = Image.open(os.path.join(args.image_folder, item["image"]))

        img_np = np.array(img)
        base64_image = encode_pil_image_to_base64(img)
        content = [
            {"type": "text", "text": "\n1. The original image:\n"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
            },
        ]

        for mask_idx, mask_rle in enumerate(item["mask_rles"]):
            mask_np = mask_utils.decode(mask_rle).astype(np.uint8)
            pil_mask = Image.fromarray((mask_np * 255).astype(np.uint8))

            assert (
                img_np.shape[:2] == mask_np.shape
            ), f"image shape mismatches with mask shape: {img_np.shape}, {mask_np.shape}"
            img_h, img_w = img_np.shape[:2]

            x0, y0, w, h = mask_to_box(mask_np)
            xc, yc = x0 + w / 2, y0 + h / 2

            # focal_crop: need to have at least min_box_w and min_box_h pixels, otherwise resizing to (384, 384) leads to artifacts that may be OOD
            w, h = max(w, args.min_box_w), max(h, args.min_box_h)
            x0, y0 = int(xc - w / 2), int(yc - h / 2)

            cropped_mask_np = mask_np[
                max(y0 - h, 0) : min(y0 + 2 * h, img_h),
                max(x0 - w, 0) : min(x0 + 2 * w, img_w),
            ]
            cropped_img_np = img_np[
                max(y0 - h, 0) : min(y0 + 2 * h, img_h),
                max(x0 - w, 0) : min(x0 + 2 * w, img_w),
            ]

            cropped_pil_img = Image.fromarray(cropped_img_np)
            cropped_pil_mask = Image.fromarray((cropped_mask_np * 255).astype(np.uint8))

            base64_cropped_image = encode_pil_image_to_base64(cropped_pil_img)
            base64_cropped_mask = encode_pil_image_to_base64(cropped_pil_mask)

            content.extend(
                [
                    {
                        "type": "text",
                        "text": f"\n{2 * mask_idx + 2}. <Prompt{mask_idx}>:\n",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_cropped_image}"
                        },
                    },
                    {
                        "type": "text",
                        "text": f"\n{2 * mask_idx + 3}. The mask of <Prompt{mask_idx}>:\n",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_cropped_mask}"
                        },
                    },
                ]
            )

        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        outputs = query(messages)
        print(outputs)
        if outputs == "True":
            true = true + 1
        item.update({"eval_result": outputs})
        output_json.append(item)

    print("Accuracy: ", true / total)
    with open(args.pred.replace(".json", "_eval.json"), "w") as f:
        json.dump(output_json, f, indent=4)
