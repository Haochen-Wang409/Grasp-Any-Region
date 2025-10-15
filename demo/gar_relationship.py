# --------------------------------------------------------
# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License")
# Grasp Any Region Project
# Written by Haochen Wang
# --------------------------------------------------------

import argparse
import ast

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, GenerationConfig

from evaluation.eval_dataset import MultiRegionDataset

TORCH_DTYPE_MAP = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference of Grasp Any Region models on DLC-Bench."
    )

    parser.add_argument(
        "--model_name_or_path",
        help="HF model name or path",
        default="ByteDance/GAR-8B",
    )
    parser.add_argument(
        "--image_path",
        help="image path",
        required=True,
    )
    parser.add_argument(
        "--mask_paths",
        help="mask path",
        required=True,
    )
    parser.add_argument(
        "--question_str",
        help="input instructions",
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

    img = Image.open(args.image_path)
    masks = []
    for mask_path in ast.literal_eval(args.mask_paths):
        mask = np.array(Image.open(mask_path).convert("L")).astype(bool)
        masks.append(mask)

    prompt_number = model.config.prompt_numbers
    prompt_tokens = [f"<Prompt{i_p}>" for i_p in range(prompt_number)] + ["<NO_Prompt>"]
    dataset = MultiRegionDataset(
        image=img,
        masks=masks,
        question_str=args.question_str
        + "\nAnswer with the correct option's letter directly.",
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


if __name__ == "__main__":
    main()
