"""
Reference: https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/eval_gpt_review.py
"""

import argparse
import json
import os
import time

import cv2
import numpy as np
import openai
import requests
from paint_util import encode_image, paint_text_box, paint_text_point
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


def get_eval(content: str, max_tokens: int):
    while True:
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful and precise assistant for checking the quality of the answer.",
                },
                {
                    "role": "user",
                    "content": content,
                },
            ]
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )
            ret = completion.choices[0].message.content
            break

        except Exception as e:
            print(e)
        time.sleep(1)

    return ret


def parse_score(review):
    try:
        score_pair = review.split("\n")[0]
        score_pair = score_pair.replace(",", " ")
        sp = score_pair.split(" ")
        print("score_pair:", score_pair, sp)
        return [float(sp[0]), float(sp[1])]
    except Exception as e:
        print(e)
        print("error", review)
        return [-1, -1]


def main(args):
    phase = args.phase  # android_QA_box
    domain = phase.split("_box")[0]  # android_QA

    if "natural" in phase:
        context_str = "The image is a natural image."
    elif "ocr" in phase:
        context_str = "The image contains text, and the user wishes to know the content of the text."
    elif "screen" in phase:
        context_str = "The image is a screenshot from a mobile phone or webpage."
    elif "panel" in phase:
        context_str = "The image is a multi-panel figure."
    elif "android" in phase:
        context_str = "The image is an andriod screenshot."
    elif "web" in phase:
        context_str = "The image is a webpage screenshot."

    question_path = f"mdvp_for_gpt4v_eval/{phase}/question.json"
    args.question = question_path
    # parser.add_argument('--question', default=question_path, help='path to question file')

    answer_list_path = [
        f"mdvp_for_gpt4v_eval/{phase}/answer.json",
        f"mdvp_for_gpt4v_eval/{phase}/prediction.json",
    ]
    args.answer_list = answer_list_path
    # parser.add_argument('--answer-list', nargs='+', default=answer_list_path, help='gpt answer and model answer json files')

    rule_path = f"annotations/rule.json"
    args.rule = rule_path
    # parser.add_argument('--rule', default=rule_path ,help='gpt rule')

    f_q = json.load(open(os.path.expanduser(args.question)))
    f_ans1 = json.load(open(os.path.expanduser(args.answer_list[0])))
    f_ans2 = json.load(open(os.path.expanduser(args.answer_list[1])))
    rule_dict = json.load(open(os.path.expanduser(args.rule), "r"))

    os.makedirs("./result", exist_ok=True)

    if os.path.isfile(os.path.expanduser(args.output)):
        cur_reviews = [
            json.loads(line) for line in open(os.path.expanduser(args.output))
        ]
    else:
        cur_reviews = []

    review_file = open(f"{args.output}", "a")

    idx = 0
    for ques, ans1, ans2 in tqdm(zip(f_q, f_ans1, f_ans2)):
        # paint som mark on image
        image_name = ques["image"]
        image_path = f"data/{domain}/images/" + image_name
        # print("loading image from {}".format(image_path))
        image = cv2.imread(image_path)
        height, width, channels = image.shape
        (width, height)
        if "bbox" in ques["annotation"]:
            bbox = ques["annotation"]["bbox"]
            paint_image_path = paint_text_box(image_path, bbox)
            rule = rule_dict["box"]
        elif "points" in ques["annotation"]:
            points = ques["annotation"]["points"]
            paint_image_path = paint_text_point(image_path, points)
            rule = rule_dict["point"]
        base64_image = encode_image(paint_image_path)

        prompt = rule["prompt"]
        role = rule["role"]
        content_text = (
            f"[Context]\{context_str}\n\n"
            f'[Question]\n{ques["text"]}\n\n'
            f'[{role} 1]\n{ans1["text"]}\n\n[End of {role} 1]\n\n'
            f'[{role} 2]\n{ans2["text"]}\n\n[End of {role} 2]\n\n'
            f"[System]\n{prompt}\n\n"
        )

        content = [
            {
                "type": "text",
                "text": content_text,
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                    "detail": "high",
                },
            },
        ]

        cur_js = {
            "id": idx + 1,
            "question_id": ques["question_id"],
            "answer1_id": ans1.get("answer_id", ans1["question_id"]),
            "answer2_id": ans2.get("answer_id", ans2["question_id"]),
            "category": phase,
        }
        # pdb.set_trace()
        if idx >= len(cur_reviews):
            review = get_eval(content, args.max_tokens)
            # print(review)

            scores = parse_score(review)
            cur_js["content"] = review
            cur_js["tuple"] = scores
            cur_js["answer1"] = ans1["text"]
            cur_js["answer2"] = ans2["text"]
            review_file.write(json.dumps(cur_js) + "\n")
            review_file.flush()
        else:
            print(f"Skipping {idx} as we already have it.")

        idx += 1
        print(idx)

    review_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChatGPT-based QA evaluation.")
    parser.add_argument(
        "--phase", help="MDVP domain", type=str, required=True
    )  # android_QA_box
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="maximum number of tokens produced in the output",
    )
    parser.add_argument(
        "--output", default=f"result/gpt_score.jsonl", help="output json dir"
    )
    args = parser.parse_args()
    main(args)
