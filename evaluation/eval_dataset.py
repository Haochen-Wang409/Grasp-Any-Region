# --------------------------------------------------------
# Copyright (2025) Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0 (the "License")
# Grasp Any Region Project
# Written by Haochen Wang
# --------------------------------------------------------

import os
import re
from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class SingleRegionCaptionDataset(Dataset):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def __init__(
        self,
        image,
        mask,
        processor,
        prompt_token="<Prompt1>",
        prompt_number=5,
        visual_prompt_tokens=[
            "<Prompt0>",
            "<Prompt1>",
            "<Prompt2>",
            "<Prompt3>",
            "<Prompt4>",
            "<NO_Prompt>",
        ],
        data_dtype=torch.bfloat16,
        **kwargs,
    ):
        self.processor = processor
        self.prompt_token = prompt_token

        self.prompt_number = prompt_number
        self.special_tokens = visual_prompt_tokens
        self.visual_prompt_ids = {
            token: self.processor.tokenizer.convert_tokens_to_ids(token) - 128256
            for token in self.special_tokens
        }

        self.image = image
        self.mask = mask
        self.data_dtype = data_dtype

    def __len__(self):
        return len(self.coco.anns)

    def _parse_annotations(self):
        image = self.image
        mask = self.mask  # binary mask

        np.array(image)
        mask_np = mask.astype(np.uint8)

        filled_matrix = -1 * np.ones((image.height, image.width), dtype=np.uint8)
        prompt_token = self.prompt_token
        prompt_id = self.visual_prompt_ids.get(
            prompt_token, self.visual_prompt_ids["<NO_Prompt>"]
        )
        assert prompt_id < 16, f"prompt_id should be less than {16}, got {prompt_id}"
        fill_area = (filled_matrix == -1) & mask_np.astype(bool)
        filled_matrix[fill_area] = prompt_id

        filled_matrix[filled_matrix == -1] = self.visual_prompt_ids["<NO_Prompt>"]

        bboxes = {}

        prompt_idx = int(re.match(r"<Prompt(\d+)>", prompt_token).group(1))
        non_zero_coords = np.argwhere(mask_np)
        y_min, x_min = non_zero_coords.min(axis=0)
        y_max, x_max = non_zero_coords.max(axis=0)
        bbox = (
            x_min / image.width,
            y_min / image.height,
            x_max / image.width,
            y_max / image.height,
        )
        bboxes[
            str(
                self.processor.tokenizer.convert_tokens_to_ids(
                    f"<|reserved_special_token_{prompt_idx + 2}|>"
                )
            )
        ] = bbox

        data_dict = {
            "image": image,
            "visual_prompt": Image.fromarray(filled_matrix),
            "bboxes": bboxes,
        }
        return data_dict

    def __getitem__(self, index):
        data_dict = deepcopy(self._parse_annotations())
        image = data_dict["image"]
        visual_prompt = data_dict["visual_prompt"]

        prompt_idx = int(re.match(r"<Prompt(\d+)>", self.prompt_token).group(1))

        # <|reserved_special_token_{idx}|> actually starts from 2
        qs = f"There are some objects I am curious about: {self.prompt_token};\n{self.prompt_token}: <|reserved_special_token_{prompt_idx + 2}|>Describe this masked region in detail."
        qs = qs.replace(
            f"<|reserved_special_token_{prompt_idx + 2}|>",
            f"<|reserved_special_token_{prompt_idx + 2}|>" * 256,
        )

        user_content = [{"type": "image", "image": image}, {"type": "text", "text": qs}]

        messages = [
            {"role": "user", "content": user_content},
        ]

        # Prepare input for model
        raw_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        model_inputs = self.processor(
            text=[raw_prompt],
            images=[image],
            visual_prompts=[visual_prompt],
            return_tensors="pt",
        )

        pixel_values = model_inputs["pixel_values"]
        mask_values = model_inputs["mask_values"]
        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        aspect_ratio = model_inputs["aspect_ratio"]

        ret = dict(
            input_ids=input_ids.cuda().unsqueeze(0),
            attention_mask=attention_mask.cuda().to(self.data_dtype).unsqueeze(0),
            pixel_values=pixel_values.cuda().to(self.data_dtype).flatten(0, 1),
            global_mask_values=mask_values.cuda().to(self.data_dtype).squeeze(),
            bboxes=[data_dict["bboxes"]],
            aspect_ratios=aspect_ratio.unsqueeze(0).cuda(),
        )
        return ret


class MultiRegionDataset(Dataset):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def __init__(
        self,
        image,
        masks,
        question_str,
        processor,
        prompt_token="<Prompt1>",
        prompt_number=5,
        visual_prompt_tokens=[
            "<Prompt0>",
            "<Prompt1>",
            "<Prompt2>",
            "<Prompt3>",
            "<Prompt4>",
            "<NO_Prompt>",
        ],
        data_dtype=torch.bfloat16,
        **kwargs,
    ):
        self.processor = processor
        self.prompt_token = prompt_token

        self.prompt_number = prompt_number
        self.special_tokens = visual_prompt_tokens
        self.visual_prompt_ids = {
            token: self.processor.tokenizer.convert_tokens_to_ids(token) - 128256
            for token in self.special_tokens
        }

        self.image = image
        self.masks = masks
        self.question_str = question_str
        self.data_dtype = data_dtype

    def __len__(self):
        return len(self.coco.anns)

    def _parse_annotations(self):
        image = self.image
        masks = self.masks  # binary mask

        width, height = image.size

        np.array(image)
        masks_np = [np.array(mask).astype(np.uint8) for mask in masks]

        for mask_id, mask in enumerate(masks_np):
            if image.width != mask.shape[1] or image.height != mask.shape[0]:
                mask = mask.resize(image.size, Image.NEAREST)
                masks[mask_id] = mask
                masks_np[mask_id] = np.array(mask).astype(np.unint8)

        prompt_matches = set(re.findall(r"<Prompt\d+>", self.question_str))
        assert len(prompt_matches) == len(masks)

        objects_desc = "There are some objects I am curious about: "
        sub_image_desc = ""
        for matched_prompt in prompt_matches:
            objects_desc += f"{matched_prompt}; "

            prompt_idx = int(re.match(r"<Prompt(\d+)>", matched_prompt).group(1))
            sub_image_desc += (
                f"{matched_prompt}: <|reserved_special_token_{prompt_idx + 2}|>\n"
            )
            sub_image_desc = sub_image_desc.replace(
                f"<|reserved_special_token_{prompt_idx + 2}|>",
                f"<|reserved_special_token_{prompt_idx + 2}|>" * 256,
            )

        prompt = objects_desc + "\n" + sub_image_desc + "\n" + self.question_str

        filled_matrix = -1 * np.ones((image.height, image.width), dtype=np.uint8)
        bboxes = {}
        for matched_prompt in prompt_matches:
            prompt_idx = int(re.match(r"<Prompt(\d+)>", matched_prompt).group(1))
            mask = masks[prompt_idx]
            prompt_token = matched_prompt
            prompt_id = self.visual_prompt_ids.get(
                prompt_token, self.visual_prompt_ids["<NO_Prompt>"]
            )
            assert (
                prompt_id < self.prompt_number + 1
            ), f"prompt_id should be less than {self.prompt_numbers + 1}, got {prompt_id}"
            fill_area = (filled_matrix == -1) & mask.astype(bool)
            filled_matrix[fill_area] = prompt_id

            non_zero_coords = np.argwhere(masks_np[mask_id])
            y_min, x_min = non_zero_coords.min(axis=0)
            y_max, x_max = non_zero_coords.max(axis=0)
            bbox = (
                x_min / image.width,
                y_min / image.height,
                x_max / image.width,
                y_max / image.height,
            )
            bboxes[
                str(
                    self.processor.tokenizer.convert_tokens_to_ids(
                        f"<|reserved_special_token_{prompt_idx + 2}|>"
                    )
                )
            ] = bbox

        filled_matrix[filled_matrix == -1] = self.visual_prompt_ids["<NO_Prompt>"]
        # convert masks to PIL.Image
        masks = [
            Image.fromarray((masks_np[i] * 255).astype(np.uint8))
            for i in range(len(masks))
        ]

        data_dict = {
            "image": image,
            "visual_prompt": Image.fromarray(filled_matrix),
            "bboxes": bboxes,
            "prompt": prompt,
        }
        return data_dict

    def __getitem__(self, index):
        data_dict = self._parse_annotations()
        image = data_dict["image"]
        visual_prompt = data_dict["visual_prompt"]
        qs = data_dict["prompt"]

        user_content = [{"type": "image", "image": image}, {"type": "text", "text": qs}]

        messages = [
            {"role": "user", "content": user_content},
        ]

        # Prepare input for model
        raw_prompt = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        model_inputs = self.processor(
            text=[raw_prompt],
            images=[image],
            visual_prompts=[visual_prompt],
            return_tensors="pt",
        )

        pixel_values = model_inputs["pixel_values"]
        mask_values = model_inputs["mask_values"]
        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        aspect_ratio = model_inputs["aspect_ratio"]

        ret = dict(
            input_ids=input_ids.cuda().unsqueeze(0),
            attention_mask=attention_mask.cuda().to(self.data_dtype).unsqueeze(0),
            pixel_values=pixel_values.cuda().to(self.data_dtype).flatten(0, 1),
            global_mask_values=mask_values.cuda().to(self.data_dtype).squeeze(),
            bboxes=[data_dict["bboxes"]],
            aspect_ratios=aspect_ratio.unsqueeze(0).cuda(),
        )
        return ret
