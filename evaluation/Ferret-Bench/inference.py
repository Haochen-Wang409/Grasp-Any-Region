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
        description="Inference of Grasp Any Region models on Ferret-Bench."
    )

    parser.add_argument(
        "--model_name_or_path",
        help="HF model name or path",
        default="ByteDance/GAR-1B",
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
        default="evaluation/Ferret-Bench/annotations/box_refer_caption.json",
    )
    parser.add_argument(
        "--image_folder",
        help="the folder of images",
        default="evaluation/Ferret-Bench/annotations/coco/val2017",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible text generation",
    )
    args = parser.parse_args()
    return args


def annToMask(ann, h, w):
    rles = mask_utils.frPyObjects(ann, h, w)
    rle = mask_utils.merge(rles)
    m = mask_utils.decode(rle)
    return m


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
    model_outputs = []
    cache_name = args.cache_name

    with open(args.anno_file, "r") as file:
        data = json.load(file)

    for idx, item in enumerate(tqdm(data)):
        image_path = os.path.join(args.image_folder, item["image"])
        img = Image.open(image_path).convert("RGB")
        width, height = img.size

        mask_r = item["annotation"]["segmentation"]
        mask = (
            annToMask(mask_r, height, width)
            if isinstance(mask_r, list)
            else mask_utils.decode(mask_r)
        )
        mask = (mask.astype(np.uint8) * 255).astype(np.uint8)

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

        model_outputs.append(
            {
                "image_path": image_path,
                "annotation": item["annotation"],
                "caption": outputs,
            }
        )

    with open(f"evaluation/Ferret-Bench/model_outputs/{cache_name}.json", "w") as file:
        json.dump(model_outputs, file, indent=4, ensure_ascii=False)

    print(f"Cache name: {cache_name}")


if __name__ == "__main__":
    main()
