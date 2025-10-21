# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2025) B-
# ytedance Inc..
# *************************************************************************

# Adapted from https://github.com/NVlabs/describe-anything/blob/main/examples/dam_with_sam.py

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
import ast

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    GenerationConfig,
    SamModel,
    SamProcessor,
)

from evaluation.eval_dataset import SingleRegionCaptionDataset

TORCH_DTYPE_MAP = dict(fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32)


def apply_sam(image, input_points=None, input_boxes=None, input_labels=None):
    inputs = sam_processor(
        image,
        input_points=input_points,
        input_boxes=input_boxes,
        input_labels=input_labels,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0][0]
    scores = outputs.iou_scores[0, 0]

    mask_selection_index = scores.argmax()

    mask_np = masks[mask_selection_index].numpy()

    return mask_np


def add_contour(img, mask, input_points=None, input_boxes=None):
    img = img.copy()

    # Draw contour
    mask = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (1.0, 1.0, 1.0), thickness=6)

    # Draw points if provided
    if input_points is not None:
        for points in input_points:  # Handle batch of points
            for x, y in points:
                # Draw a filled circle for each point
                cv2.circle(
                    img,
                    (int(x), int(y)),
                    radius=10,
                    color=(1.0, 0.0, 0.0),
                    thickness=-1,
                )
                # Draw a white border around the circle
                cv2.circle(
                    img, (int(x), int(y)), radius=10, color=(1.0, 1.0, 1.0), thickness=2
                )

    # Draw boxes if provided
    if input_boxes is not None:
        for box_batch in input_boxes:  # Handle batch of boxes
            for box in box_batch:  # Iterate through boxes in the batch
                x1, y1, x2, y2 = map(int, box)
                # Draw rectangle with white color
                cv2.rectangle(
                    img, (x1, y1), (x2, y2), color=(1.0, 1.0, 1.0), thickness=4
                )
                # Draw inner rectangle with red color
                cv2.rectangle(
                    img, (x1, y1), (x2, y2), color=(1.0, 0.0, 0.0), thickness=2
                )

    return img


def denormalize_coordinates(coords, image_size, is_box=False):
    """Convert normalized coordinates (0-1) to pixel coordinates."""
    width, height = image_size
    if is_box:
        # For boxes: [x1, y1, x2, y2]
        x1, y1, x2, y2 = coords
        return [int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)]
    else:
        # For points: [x, y]
        x, y = coords
        return [int(x * width), int(y * height)]


def print_streaming(text):
    """Helper function to print streaming text with flush"""
    print(text, end="", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detailed Localized Image Descriptions with SAM"
    )
    parser.add_argument(
        "--model_name_or_path",
        help="HF model name or path",
        default="HaochenWang/GAR-8B",
    )
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the image file"
    )
    parser.add_argument(
        "--points",
        type=str,
        default="[[1172, 812], [1572, 800]]",
        help="List of points for SAM input",
    )
    parser.add_argument(
        "--box",
        type=str,
        default="[773, 518, 1172, 812]",
        help="Bounding box for SAM input (x1, y1, x2, y2)",
    )
    parser.add_argument(
        "--use_box",
        action="store_true",
        help="Use box instead of points for SAM input (default: use points)",
    )
    parser.add_argument(
        "--normalized_coords",
        action="store_true",
        help="Interpret coordinates as normalized (0-1) values",
    )
    parser.add_argument(
        "--output_image_path",
        type=str,
        default=None,
        help="Path to save the output image with contour",
    )
    parser.add_argument(
        "--data_type",
        help="data dtype",
        type=str,
        choices=["fp16", "bf16", "fp32"],
        default="bf16",
    )

    args = parser.parse_args()
    data_dtype = TORCH_DTYPE_MAP[args.data_type]

    # Load the image
    img = Image.open(args.image_path).convert("RGB")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    image_size = img.size  # (width, height)

    # Prepare input_points or input_boxes
    if args.use_box:
        input_boxes = ast.literal_eval(args.box)
        if args.normalized_coords:
            input_boxes = denormalize_coordinates(input_boxes, image_size, is_box=True)
        input_boxes = [[input_boxes]]  # Add an extra level of nesting
        print(f"Using input_boxes: {input_boxes}")
        mask_np = apply_sam(img, input_boxes=input_boxes)
    else:
        input_points = ast.literal_eval(args.points)
        if args.normalized_coords:
            input_points = [
                denormalize_coordinates(point, image_size) for point in input_points
            ]
        # Assume all points are foreground
        input_labels = [1] * len(input_points)
        input_points = [[x, y] for x, y in input_points]  # Convert to list of lists
        input_points = [input_points]  # Wrap in outer list
        input_labels = [input_labels]  # Wrap labels in list
        print(f"Using input_points: {input_points}")
        mask_np = apply_sam(img, input_points=input_points, input_labels=input_labels)

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

    # Get description
    prompt_number = model.config.prompt_numbers
    prompt_tokens = [f"<Prompt{i_p}>" for i_p in range(prompt_number)] + ["<NO_Prompt>"]
    dataset = SingleRegionCaptionDataset(
        image=img,
        mask=mask_np,
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

    if args.output_image_path:
        img_np = np.asarray(img).astype(float) / 255.0

        # Prepare visualization inputs
        vis_points = input_points if not args.use_box else None
        vis_boxes = input_boxes if args.use_box else None

        img_with_contour_np = add_contour(
            img_np, mask_np, input_points=vis_points, input_boxes=vis_boxes
        )
        img_with_contour_pil = Image.fromarray(
            (img_with_contour_np * 255.0).astype(np.uint8)
        )
        img_with_contour_pil.save(args.output_image_path)
        print(f"Output image with contour saved as {args.output_image_path}")
