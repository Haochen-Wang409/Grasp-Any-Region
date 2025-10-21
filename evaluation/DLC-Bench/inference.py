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
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor, GenerationConfig

from evaluation.eval_dataset import SingleRegionCaptionDataset

TORCH_DTYPE_MAP = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference of Grasp Any Region models on DLC-Bench."
    )

    parser.add_argument(
        "--model_name_or_path",
        help="HF model name or path",
        default="HaochenWang/GAR-1B",
    )
    parser.add_argument(
        "--cache_name",
        help="cache name to save model outputs.",
        default="gar_1b",
    )
    parser.add_argument(
        "--data_type",
        help="data dtype",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        default="bf16",
    )
    parser.add_argument(
        "--anno_file",
        help="path to the annotation file.",
        default="evaluation/DLC-Bench/annotations/annotations.json",
    )
    parser.add_argument(
        "--image_folder",
        help="the folder of images",
        default="evaluation/DLC-Bench/annotations",
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
    )
    model.cuda()
    model.eval()

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    model_outputs = {}
    cache_name = args.cache_name

    # This coco instance is actually an o365 subset. This is for code reuse.
    coco = COCO(args.anno_file)
    img_ids = list(coco.imgs.keys())
    num_anns = len(coco.anns)
    pbar = tqdm(total=num_anns)

    for img_id in img_ids:
        ann_ids = select_ann(coco, img_id)
        img_info = coco.loadImgs(img_id)[0]

        for i, ann_id in enumerate(ann_ids):
            if ann_id in model_outputs.keys():
                pbar.update(1)
                continue

            anns = coco.loadAnns([ann_id])
            mask = coco.annToMask(anns[0])

            img_path = os.path.join(args.image_folder, "images", img_info["file_name"])
            img = Image.open(img_path)

            prompt_number = model.config.prompt_numbers
            prompt_tokens = [f"<Prompt{i_p}>" for i_p in range(prompt_number)] + [
                "<NO_Prompt>"
            ]
            dataset = SingleRegionCaptionDataset(
                image=img,
                mask=mask,
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
                generate_ids.sequences[0], skip_special_tokens=True
            ).strip()

            print(outputs)  # Print model output for this image

            model_outputs[ann_id] = outputs
            pbar.update(1)
    pbar.close()

    with open(f"evaluation/DLC-Bench/model_outputs/{cache_name}.json", "w") as file:
        json.dump(model_outputs, file, indent=4, ensure_ascii=False)

    print(f"Cache name: {cache_name}")


if __name__ == "__main__":
    main()
