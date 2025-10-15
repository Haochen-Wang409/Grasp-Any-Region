import base64
import copy
import io
import json
import math
import os
import random
import re

import numpy as np
import pycocotools.mask as mask_util
import torch
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import Dataset
from transformers.image_utils import PILImageResampling

from ..models.modeling.image_processing_perception_lm_fast import (
    PerceptionLMImageProcessorFast,
)
from ..models.modeling.processing_perception_lm import PerceptionLMProcessor

prompt_list = [
    "Describe the masked region {prompt_suffix}.",
    "Describe the masked area {prompt_suffix}.",
    "What can you describe about the masked region {prompt_suffix}?",
    "Can you describe the masked region {prompt_suffix}?",
    "Provide an explanation of the masked region {prompt_suffix}.",
    "Depict the masked area {prompt_suffix}.",
    "Portray the masked area {prompt_suffix}.",
    "Describe what the masked region looks like {prompt_suffix}.",
    "Illustrate the masked region {prompt_suffix}.",
    "How would you explain the masked area {prompt_suffix}?",
    "What details can you provide about the masked region {prompt_suffix}?",
    "What does the masked region entail {prompt_suffix}?",
    "How would you illustrate the masked region {prompt_suffix}?",
    "How would you depict the masked area {prompt_suffix}?",
    "How would you portray the masked area {prompt_suffix}?",
    "Give a detailed description of the masked region.",
    "Provide a thorough description of the masked region.",
    "Can you explain the details of the masked area?",
    "Give a detailed account of the masked region.",
    "Describe the masked area comprehensively.",
    "Provide an in-depth description of the masked region.",
    "Explain the specifics of the masked area.",
    "Can you provide a thorough explanation of the masked region?",
    "What are the details of the masked area?",
    "Provide a comprehensive description of the masked area.",
    "What specific details can you provide about the masked region?",
    "Can you give an in-depth account of the masked section?",
    "What are the main characteristics of the masked region?",
    "Give a thorough description of the masked area's details.",
    "Provide detailed information about the masked area.",
]


def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 768 * 768,
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions are divisible by 'factor'.
    2. The total number of pixels is within ['min_pixels', 'max_pixels'].
    3. The aspect ratio is preserved as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(
            f"height:{height} or width:{width} must be larger than factor:{factor}"
        )
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


class GraspAnyRegionDataset(Dataset):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def __init__(
        self,
        pano_jsons,
        model_path,
        special_tokens=None,
        dynamic_image_size=True,
        repeats=1,
        max_num_tiles=16,
        prompt_augmentation=False,
        prompt_numbers=5,
        **kwargs,
    ):
        self._system = ""
        self.repeats = repeats
        self.dynamic_image_size = dynamic_image_size
        self.max_num_tiles = max_num_tiles if dynamic_image_size else 1
        self.prompt_augmentation = prompt_augmentation
        self.prompt_numbers = prompt_numbers

        self.pano_jsons = pano_jsons

        self.processor = PerceptionLMProcessor.from_pretrained(model_path)
        image_processor_config = self.processor.image_processor.__dict__
        image_processor_config.pop("_processor_class", None)

        self.processor.image_processor = PerceptionLMImageProcessorFast.from_dict(
            image_processor_config
        )
        self.processor.image_processor.max_num_tiles = self.max_num_tiles

        self.processor_mask = PerceptionLMProcessor.from_pretrained(model_path)
        self.processor_mask.image_processor = PerceptionLMImageProcessorFast.from_dict(
            image_processor_config
        )
        self.processor_mask.image_processor.max_num_tiles = self.max_num_tiles
        self.processor_mask.image_processor.resample = PILImageResampling.NEAREST

        if special_tokens is not None:
            self.special_tokens = special_tokens
            self.processor.tokenizer.add_tokens(special_tokens, special_tokens=True)
            self.processor_mask.tokenizer.add_tokens(
                special_tokens, special_tokens=True
            )
            self.visual_prompt_ids = {
                token: self.processor.tokenizer.convert_tokens_to_ids(token) - 128256
                for token in special_tokens
            }

        self.datas, self.data_lengths = self.read_pano_json()
        # self.max_length = max_length
        self._max_refetch = 1000

        self.tcs_loader = None

        print(
            "GraspAnyRegion dataset, include {} items.".format(sum(self.data_lengths))
        )

    def prompt_aug(self, caption):
        # following DAM paper.
        random_number = random.random()

        if random_number < 0.6:  # default in detail, select from either set
            prompt_index = random.randint(0, 29)
            prompt = prompt_list[prompt_index]
            if prompt_index < 15:
                prompt = prompt.replace("{prompt_suffix}", "in detail")

        elif random_number > 0.8:  # caption word count
            caption_word_count = len(caption.split())

            prompt_index = random.randint(0, 14)
            prompt = prompt_list[prompt_index]
            if caption_word_count < 10:
                prompt = prompt.replace(
                    "{prompt_suffix}", f"in {caption_word_count} words"
                )
            elif caption_word_count > 200:
                prompt = prompt.replace("{prompt_suffix}", f"in more than 200 words")
            else:
                count_nearest_ten = round(caption_word_count / 10) * 10
                word = random.choice(["about", "around"])
                prompt = prompt.replace(
                    "{prompt_suffix}", f"in {word} {count_nearest_ten} words"
                )

        else:  # sentences count
            sentences = re.split(r"[.!?]", caption)
            sentences_count = len([s for s in sentences if s.strip()])
            prompt_index = random.randint(0, 14)
            prompt = prompt_list[prompt_index]
            if sentences_count == 1:
                prompt = prompt.replace(
                    "{prompt_suffix}",
                    random.choice(
                        ["in a sentence", "in one sentence", "in 1 sentence"]
                    ),
                )
            else:
                prompt = prompt.replace(
                    "{prompt_suffix}", f"in {sentences_count} sentences"
                )

        return prompt

    @property
    def modality_length(self):
        length_list = []
        for idx in range(sum(self.data_lengths)):
            length_list.append(100)
        return length_list * self.repeats

    def __len__(self):
        return sum(self.data_lengths) * self.repeats

    def read_pano_json(self):
        all_json_info = []
        all_json_length = []
        for pano_json in self.pano_jsons:
            if pano_json.endswith(".json"):
                with open(pano_json, "r") as f:
                    json_info = json.load(f)
            else:
                json_info = load_from_disk(pano_json)

            all_json_info.append(json_info)
            all_json_length.append(len(json_info))
            print(f"=> Loaded {pano_json} with {len(json_info)} items.")

        print(f"Total data counts: {sum(all_json_length)}")
        return all_json_info, all_json_length

    def sort_masks_by_area(self, masks):
        areas = []
        for mask in masks:
            area = np.sum(mask)
            areas.append(area)
        indexes = np.argsort(np.array(areas))[
            ::-1
        ]  # sort the mask from large area to small area
        return indexes

    def _parse_annotations(self, ann_info):
        captions = []
        for conv in ann_info["conversations"]:
            if conv["from"] == "gpt":
                captions.append(conv["value"])

        image_path = ann_info["image"]
        if image_path is not None:
            if isinstance(image_path, Image.Image):
                image = image_path
            elif image_path.startswith("data:base64,"):
                base64_str = image_path.replace("data:base64,", "")
                image_bytes = base64.b64decode(base64_str)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")

            if ann_info.get("mask_rle", None) is not None:
                mask_caption_data = True
                if isinstance(ann_info["mask_rle"], list):
                    masks = [
                        mask_util.decode(rle_dict) for rle_dict in ann_info["mask_rle"]
                    ]
                elif isinstance(ann_info["mask_rle"], dict):
                    masks = [mask_util.decode(ann_info["mask_rle"])]
                else:
                    raise ValueError(
                        f"mask_rle should be list or dict, but got {type(ann_info['mask_rle'])}"
                    )
            else:
                # all 1 mask
                mask_caption_data = False
                masks = [np.ones((image.height, image.width), dtype=np.uint8)] * len(
                    captions
                )
        else:
            print("no image, skip.")
            return None

        masks_np = [np.array(mask).astype(np.uint8) for mask in masks]
        bboxes = {}

        for mask_id, mask in enumerate(masks):
            if image.width != mask.shape[1] or image.height != mask.shape[0]:
                mask = mask.resize(image.size, Image.NEAREST)
                masks[mask_id] = mask
                masks_np[mask_id] = np.array(mask).astype(np.unint8)

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
                        f"<|reserved_special_token_{mask_id + 2}|>"
                    )
                )
            ] = bbox

        # random sampling used prompt indexes
        prompt_indexes = [i_p for i_p in range(self.prompt_numbers)]
        random.shuffle(prompt_indexes)
        num_selected = min(len(masks_np), self.prompt_numbers - 1)
        selected_prompt_indexes = prompt_indexes[:num_selected]
        selected_prompt_tokens = [f"<Prompt{i_p}>" for i_p in selected_prompt_indexes]
        selected_prompt_img_tokens = [
            f"<|reserved_special_token_{i_p+2}|>" for i_p in selected_prompt_indexes
        ]
        # for none prompt
        none_prompt = True
        not_selected_prompt_indexes = prompt_indexes[num_selected:]
        not_selected_prompt_tokens = [
            f"<Prompt{i_p}>" for i_p in not_selected_prompt_indexes
        ]
        not_selected_prompt_img_tokens = [
            f"<|reserved_special_token_{i_p+2}|>" for i_p in not_selected_prompt_indexes
        ]

        if not mask_caption_data:
            filled_matrix = self.visual_prompt_ids[selected_prompt_tokens[0]] * np.ones(
                (image.height, image.width), dtype=np.uint8
            )
            ret = {
                "masks": masks,
                "bboxes": bboxes,
                "conversations": ann_info["conversations"],
                "image": image,
                "visual_prompt_matrix": Image.fromarray(filled_matrix),
                "mask_caption_data": False,
            }
            return ret

        prompt_str = ""
        for conv in ann_info["conversations"]:
            prompt_str += f"\n{conv['value']}"
        prompt_matches = set(re.findall(r"<Prompt\d+>", prompt_str))

        if len(prompt_matches) == 0:
            # build visual_prompt list for each mask
            crop_phrases = []
            for idx in range(num_selected):
                prompt_id = self.visual_prompt_ids.get(
                    selected_prompt_tokens[idx], self.visual_prompt_ids["<NO_Prompt>"]
                )

                crop_phrases.append(f"{selected_prompt_tokens[idx]}")

            # modify conversations
            conversation = []
            ret_masks = []
            first_question_merged = False
            if len(crop_phrases) > 0 and len(masks) > 0:
                # 组合对象列表与第一个问题
                objects_desc = (
                    "There are some objects I am curious about: "
                    + "; ".join(crop_phrases)
                    + "; "
                )
                if self.prompt_augmentation:
                    prompt = self.prompt_aug(captions[0])
                else:
                    prompt = "Describe this masked region in detail."
                first_question = f"{selected_prompt_tokens[0]}: {selected_prompt_img_tokens[0]}{prompt}"
                first_question = first_question.replace(
                    selected_prompt_img_tokens[0], selected_prompt_img_tokens[0] * 256
                )

                conversation.append(
                    {
                        "from": "human",
                        "value": objects_desc + "\n" + first_question,
                    }
                )
                first_question_merged = True

            # 处理剩余对话
            for i in range(num_selected):
                mask = masks[i]
                obj_description = captions[i]
                if i == 0 and first_question_merged:
                    # directly add answer for the first qustion
                    conversation.append({"from": "gpt", "value": obj_description})
                    ret_masks.append(mask)
                    continue
                if none_prompt and random.random() < 0.05:
                    question = f"{not_selected_prompt_tokens[0]}: {self.prompt_aug(obj_description)}"
                    conversation.append({"from": "human", "value": question})
                    conversation.append(
                        {
                            "from": "gpt",
                            "value": f"{not_selected_prompt_tokens[0]} is not in the image.",
                        }
                    )
                    none_prompt = False

                if self.prompt_augmentation:
                    prompt = self.prompt_aug(obj_description)
                else:
                    prompt = "Describe this masked region in detail."

                question = f"{selected_prompt_tokens[i]}: {selected_prompt_img_tokens[i]}{prompt}"
                question = question.replace(
                    selected_prompt_img_tokens[i], selected_prompt_img_tokens[i] * 256
                )
                conversation.append({"from": "human", "value": question})
                conversation.append({"from": "gpt", "value": f"{obj_description}"})
                ret_masks.append(mask)

            filled_matrix = -1 * np.ones((image.height, image.width), dtype=np.uint8)
            bboxes = {}
            for i in range(num_selected):
                mask = masks[i]
                prompt_token = selected_prompt_tokens[i]
                prompt_id = self.visual_prompt_ids.get(
                    prompt_token, self.visual_prompt_ids["<NO_Prompt>"]
                )
                assert (
                    prompt_id < self.prompt_numbers + 1
                ), f"prompt_id should be less than {self.prompt_numbers + 1}, got {prompt_id}"
                fill_area = (filled_matrix == -1) & mask.astype(bool)
                filled_matrix[fill_area] = prompt_id

                prompt_idx = int(re.match(r"<Prompt(\d+)>", prompt_token).group(1))

                non_zero_coords = np.argwhere(np.array(mask))
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
                for i in range(num_selected)
            ]

        else:
            # modify the first conversations
            conversation = copy.deepcopy(ann_info["conversations"])
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

            conversation[0]["value"] = (
                objects_desc + "\n" + sub_image_desc + "\n" + conversation[0]["value"]
            )

            new_masks_np = []
            filled_matrix = -1 * np.ones((image.height, image.width), dtype=np.uint8)
            for matched_prompt in prompt_matches:
                prompt_idx = int(re.match(r"<Prompt(\d+)>", matched_prompt).group(1))
                mask = masks[prompt_idx]
                prompt_token = matched_prompt
                prompt_id = self.visual_prompt_ids.get(
                    prompt_token, self.visual_prompt_ids["<NO_Prompt>"]
                )
                assert (
                    prompt_id < self.prompt_numbers + 1
                ), f"prompt_id should be less than {self.prompt_numbers + 1}, got {prompt_id}"
                fill_area = (filled_matrix == -1) & mask.astype(bool)
                filled_matrix[fill_area] = prompt_id
                new_masks_np.append(np.array(mask).astype(np.uint8))

            filled_matrix[filled_matrix == -1] = self.visual_prompt_ids["<NO_Prompt>"]
            masks_np = copy.deepcopy(new_masks_np)
            # convert masks to PIL.Image
            masks = [
                Image.fromarray((mask_np * 255).astype(np.uint8))
                for mask_np in masks_np
            ]

        ret = {
            "masks": masks,
            "bboxes": bboxes,
            "conversations": conversation,
            "image": image,
            "visual_prompt_matrix": Image.fromarray(filled_matrix),
            "mask_caption_data": "mask_rle" in ann_info.keys(),
        }
        return ret

    def parse_label(self, labels):
        start_tokens = torch.tensor([128006, 78191, 128007, 271], device=labels.device)
        end_token = 128009

        labels = labels.clone()
        mask = torch.full_like(labels, fill_value=-100)

        i = 0
        while i < len(labels):
            if i + len(start_tokens) <= len(labels) and torch.equal(
                labels[i : i + len(start_tokens)], start_tokens
            ):
                start = i + len(start_tokens)
                try:
                    end = (labels[start:] == end_token).nonzero(as_tuple=True)[0][
                        0
                    ].item() + start
                except IndexError:
                    break
                # keep [start:end+1]
                if end >= start:
                    mask[start : end + 1] = labels[start : end + 1]
                i = end + 1
            else:
                i += 1

        return mask

    def prepare_data(self, index, **kwargs):
        index = index % sum(self.data_lengths)

        def find_dataset_index(index, data_lengths):
            cumulative = 0
            for i, length in enumerate(data_lengths):
                if index < cumulative + length:
                    return i, index - cumulative
                cumulative += length

        data_idx, internal_index = find_dataset_index(index, self.data_lengths)

        data_dict = copy.deepcopy(self.datas[data_idx][internal_index])
        data_dict = self._parse_annotations(data_dict)

        if data_dict is None:
            return None

        image = data_dict["image"]
        convs = data_dict["conversations"]
        visual_prompt = data_dict["visual_prompt_matrix"]

        w, h = image.size
        if w < 10 or h < 10:
            return None

        if data_dict["mask_caption_data"]:
            messages, messages_mask = [], []
            for i, conv in enumerate(convs):
                if i == 0:
                    assert conv["from"] == "human"
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {
                                    "type": "text",
                                    "text": conv["value"].replace("<image>\n", ""),
                                },
                            ],
                        },
                    )
                    messages_mask.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": visual_prompt},
                                {
                                    "type": "text",
                                    "text": conv["value"].replace("<image>\n", ""),
                                },
                            ],
                        },
                    )
                    continue

                assert "<image>" not in conv["value"]
                if conv["from"] == "human":
                    messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": conv["value"]}],
                        }
                    )
                    messages_mask.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": conv["value"]}],
                        }
                    )
                elif conv["from"] == "gpt":
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": conv["value"]}],
                        }
                    )
                    messages_mask.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": conv["value"]}],
                        }
                    )
                else:
                    raise NotImplementedError
        else:
            # keep the same with the original provided conversation
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": data_dict["conversations"][0]["value"].replace(
                                "<image>\n", ""
                            ),
                        },
                    ],
                },
            ]
            messages_mask = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": visual_prompt},
                        {
                            "type": "text",
                            "text": data_dict["conversations"][0]["value"].replace(
                                "<image>\n", ""
                            ),
                        },
                    ],
                },
            ]
            for conv in data_dict["conversations"][1:]:
                assert "<image>" not in conv["value"]
                if conv["from"] == "human":
                    messages.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": conv["value"]}],
                        }
                    )
                    messages_mask.append(
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": conv["value"]}],
                        }
                    )
                elif conv["from"] == "gpt":
                    messages.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": conv["value"]}],
                        }
                    )
                    messages_mask.append(
                        {
                            "role": "assistant",
                            "content": [{"type": "text", "text": conv["value"]}],
                        }
                    )

        try:
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )

            inputs_mask = self.processor.apply_chat_template(
                messages_mask,
                add_generation_prompt=False,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
            )
        except:
            print("tokenization failed.")
            return None

        pixel_values = inputs["pixel_values"]
        aspect_ratio = inputs["aspect_ratio"]
        mask_values = inputs_mask["pixel_values"]
        input_ids = inputs["input_ids"].squeeze(0)
        try:
            assert torch.equal(inputs["input_ids"], inputs_mask["input_ids"])
            assert torch.equal(inputs["attention_mask"], inputs_mask["attention_mask"])
        except:
            print("inputs are different, skip")
            return None

        labels = inputs["input_ids"].squeeze(0).clone()
        labels = self.parse_label(labels)
        attention_mask = inputs["attention_mask"].squeeze(0)

        ret = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            global_mask_values=mask_values,
            aspect_ratio=aspect_ratio.unsqueeze(0),
            bboxes=data_dict["bboxes"],
        )
        return ret

    def _rand_another(self):
        idx = random.randint(0, sum(self.data_lengths))
        return idx

    def __getitem__(self, index):
        for _ in range(self._max_refetch + 1):
            try:
                data = self.prepare_data(index, padding=False, return_tensors="pt")
            except:
                data = None

            if data is None:
                index_old = index
                index = self._rand_another()
                print(f"[WARNING] data {index_old} is None, use {index}!")
                continue
            return data
