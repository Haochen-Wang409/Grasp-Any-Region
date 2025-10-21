# --------------------------------------------------------
# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License")
# Grasp Any Region Project
# Written by Haochen Wang
# --------------------------------------------------------

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, GenerationConfig

from evaluation.eval_dataset import MultiRegionDataset

TORCH_DTYPE_MAP = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference of Grasp Any Region models on GAR-Bench."
    )

    parser.add_argument(
        "--model_name_or_path",
        help="HF model name or path",
        default="HaochenWang/GAR-8B",
    )
    parser.add_argument(
        "--cache_name",
        help="cache name for saving results",
        type=str,
        default="gar_8b",
    )
    parser.add_argument(
        "--anno_file",
        help="annotation file path",
        required=True,
    )
    parser.add_argument(
        "--image_folder",
        help="the folder of images",
        default="evaluation/GAR-Bench/annotations",
    )
    parser.add_argument(
        "--mode",
        help="mode to build questions",
        type=str,
        choices=["vqa", "simple", "detailed"],
        required=True,
    )
    parser.add_argument(
        "--data_type",
        help="data dtype",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        default="bf16",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible text generation",
    )
    args = parser.parse_args()
    return args


def select_ann(coco, img_id, area_min=None, area_max=None):
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


def main():
    args = parse_args()
    data_dtype = TORCH_DTYPE_MAP[args.data_type]
    torch.manual_seed(args.seed)

    # init ditribution for dispatch_modules in LLM
    torch.cuda.set_device(0)
    torch.distributed.init_process_group(backend="nccl")

    # build HF model
    model = AutoModel.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=data_dtype,
        device_map="cuda:0",
    ).eval()

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )

    model_outputs = []
    cache_name = args.cache_name

    with open(args.anno_file, "r") as file:
        data = json.load(file)

    for item in tqdm(data):
        img = Image.open(os.path.join(args.image_folder, item["image"]))

        # build question for different mode
        if args.mode == "vqa":
            question_str = f"Question: {item['question']}\nOptions:"
            for op in item["choices"]:
                question_str += f"\n{op}"
            question_str += "\nAnswer with the correct option's letter directly."
        elif args.mode == "simple":
            question_str = item["question"]
        elif args.mode == "detailed":
            question_str = "Describe <Prompt0> in detail, including the relationship with <Prompt1>."
        else:
            raise NotImplementedError

        masks = []
        for mask_idx, mask_rle in enumerate(item["mask_rles"]):
            mask_np = mask_utils.decode(mask_rle).astype(np.uint8)
            masks.append((mask_np * 255).astype(np.uint8))

        prompt_number = model.config.prompt_numbers
        prompt_tokens = [f"<Prompt{i_p}>" for i_p in range(prompt_number)] + [
            "<NO_Prompt>"
        ]
        dataset = MultiRegionDataset(
            image=img,
            masks=masks,
            question_str=question_str,
            processor=processor,
            prompt_number=prompt_number,
            visual_prompt_tokens=prompt_tokens,
            data_dtype=data_dtype,
        )

        data_sample = dataset[0]

        with torch.no_grad():
            generate_ids = model.generate(
                **data_sample,
                generation_config=GenerationConfig(
                    max_new_tokens=1024,
                    do_sample=False,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                ),
                return_dict=True,
            )

        outputs = processor.tokenizer.decode(
            generate_ids.sequences[0], skip_special_tokens=False
        ).strip()
        if outputs.endswith("<|eot_id|>"):
            outputs = outputs.replace("<|eot_id|>", "")
        print(outputs)

        item["model_output"] = outputs
        model_outputs.append(item)

    cache_name += f"_{args.mode}"
    print(f"Cache name: {cache_name}")

    with open(f"evaluation/GAR-Bench/model_outputs/{cache_name}.json", "w") as file:
        json.dump(model_outputs, file, indent=4, ensure_ascii=False)

    if args.mode == "vqa":
        # directly compute accuracy using exact-matching
        for category in set([x["type"] for x in model_outputs]):
            results = [x for x in model_outputs if x["type"] == category]
            total = len(results)
            correct = len(
                [x for x in results if x["model_output"].lower() == x["answer"].lower()]
            )
            print(f"{category}: [{correct}/{total}]={round(correct / total * 100, 1)}")

        total = len(model_outputs)
        correct = len(
            [
                x
                for x in model_outputs
                if x["model_output"].lower() == x["answer"].lower()
            ]
        )
        print(f"=> overall: [{correct}/{total}]={round(correct / total * 100, 1)}")


if __name__ == "__main__":
    main()
