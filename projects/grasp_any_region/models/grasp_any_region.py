from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training
from transformers import AutoConfig, AutoProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast
from xtuner.model.modules import dispatch_modules
from xtuner.model.utils import (
    find_all_linear_names,
    get_peft_model_state_dict,
    guess_load_checkpoint,
    make_inputs_require_grad,
    traverse_dict,
)
from xtuner.registry import BUILDER

from .modeling.modeling_perception_lm import PerceptionLMForConditionalGeneration


class GraspAnyRegion(BaseModel):
    def __init__(
        self,
        mllm,
        freeze_llm=False,
        freeze_visual_encoder=False,
        freeze_connector=False,
        unfreeze_vocab=False,
        unfreeze_lm_head=False,
        llm_lora=None,
        pretrained_pth=None,
        use_activation_checkpointing=True,
        vocab_embeds_name="tok_embeddings",
        lm_head_name="output",
        prompt_numbers=15,
    ):
        super().__init__()

        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.freeze_connector = freeze_connector
        self.unfreeze_vocab = unfreeze_vocab
        self.unfreeze_lm_head = unfreeze_lm_head
        self.use_llm_lora = llm_lora is not None
        self.use_activation_checkpointing = use_activation_checkpointing
        self.vocab_embeds_name = vocab_embeds_name
        self.lm_head_name = lm_head_name
        self.prompt_numbers = prompt_numbers

        config = AutoConfig.from_pretrained(
            mllm["pretrained_model_name_or_path"], trust_remote_code=True
        )

        self.config = config

        traverse_dict(mllm)

        self.model = PerceptionLMForConditionalGeneration.from_pretrained(
            mllm["pretrained_model_name_or_path"], trust_remote_code=True
        )

        # build mask_patch_embedding
        patch_size_h = (
            self.model.config.vision_config.model_args["img_size"][0]
            // self.model.config.vision_config.model_args["ref_feat_shape"][0]
        )
        patch_size_w = (
            self.model.config.vision_config.model_args["img_size"][1]
            // self.model.config.vision_config.model_args["ref_feat_shape"][1]
        )
        kernel_size = [patch_size_h, patch_size_w]
        self.mask_patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=self.model.config.vision_config.num_features,
            kernel_size=kernel_size,
            stride=kernel_size,
            bias=False,
        )
        # zero-init
        for param in self.mask_patch_embedding.parameters():
            nn.init.zeros_(param)

        self.model.model.config.use_cache = False

        dispatch_modules(self.model.model)

        self.processor = AutoProcessor.from_pretrained(
            mllm["pretrained_model_name_or_path"], trust_remote_code=True
        )

        if self.freeze_llm:
            self.model.model.language_model.requires_grad_(False)

        if self.freeze_visual_encoder:
            self.model.model.vision_tower.requires_grad_(False)

        if self.freeze_connector:
            self.model.model.multi_modal_projector.requires_grad_(False)

        if use_activation_checkpointing:
            # it is necessary when using gradient checkpointing
            if hasattr(self.model.model, "enable_input_require_grads"):
                self.model.model.enable_input_require_grads()
            else:
                self.model.model.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )

        self._add_special_tokens()
        self.gradient_checkpointing_enable()

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora)

        # put this after llm_lora
        if self.unfreeze_vocab:
            self.model.get_input_embeddings().requires_grad_(True)
        if self.unfreeze_lm_head:
            self.model.get_output_embeddings().requires_grad_(True)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            msg = self.load_state_dict(
                pretrained_state_dict, strict=False
            )  # TODO, check whether the internvl2 weights are loaded correctly.
            print(f"Load pretrained weight from {pretrained_pth} with msg: {msg}")

        self._count = 0
        print_log(self, logger="current")
        print_log("Perception_LM construction is complete", logger="current")

    def _add_special_tokens(self):
        assert hasattr(self, "processor")

        visual_prompt_nums = self.prompt_numbers
        visual_prompt_tokens = [f"<Prompt{i}>" for i in range(visual_prompt_nums)]
        visual_prompt_tokens.append("<NO_Prompt>")
        special_tokens = visual_prompt_tokens
        num_new_tokens = self.processor.tokenizer.add_tokens(
            special_tokens, special_tokens=True
        )
        self.model.resize_token_embeddings(len(self.processor.tokenizer))
        print_log(f"Added {num_new_tokens} special tokens.")

    def _parse_lora_config(self, lora_config):
        if (
            isinstance(lora_config, dict)
            or isinstance(lora_config, Config)
            or isinstance(lora_config, ConfigDict)
        ):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self, lora_config, use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.model.model = prepare_model_for_kbit_training(
            self.model.model, use_activation_checkpointing
        )
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.model.model)
            lora_config.target_modules = modules

        self.model.model = get_peft_model(self.model.model, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.model.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.model.model.gradient_checkpointing_disable()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()

        to_return.update(
            {
                k: v
                for k, v in state_dict.items()
                if "tok_embeddings" in k or "embed" in k or "embed_tokens" in k
            }
        )
        # logit head
        to_return.update(
            {
                k: v
                for k, v in state_dict.items()
                if "output." in k and "llm" in k and "lora" not in k
            }
        )
        to_return.update(
            {k: v for k, v in state_dict.items() if "lm_head" in k and "lora" not in k}
        )
        to_return.update(
            {k: v for k, v in state_dict.items() if "output" in k and "lora" not in k}
        )

        # Step 1. visual_encoder
        if not self.freeze_visual_encoder:
            to_return.update(
                {k: v for k, v in state_dict.items() if "model.visual." in k}
            )
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.model.model, state_dict=state_dict)
            )
        elif not self.freeze_llm:
            to_return.update({k: v for k, v in state_dict.items() if "model.model."})

        # Step 3. mask_patch_embedding
        to_return.update(
            {k: v for k, v in state_dict.items() if "mask_patch_embedding." in k}
        )
        to_return.update({k: v for k, v in state_dict.items() if "mask_conv." in k})

        return to_return

    def init_weights(self):
        pass

    def _merge(self, tiles: torch.Tensor, ncw: int, nch: int) -> torch.Tensor:
        batch_size, num_tiles, num_channels, tile_height, tile_width = tiles.size()
        assert num_tiles == ncw * nch, f"{ncw * nch} != {num_tiles}"

        tiles = tiles.view(batch_size, nch, ncw, num_channels, tile_height, tile_width)
        tiles = tiles.permute(0, 3, 1, 4, 2, 5).contiguous()

        original_height = nch * tile_height
        original_width = ncw * tile_width

        image = tiles.view(batch_size, num_channels, original_height, original_width)

        return image

    def forward(self, data, data_samples=None, mode="loss"):
        crop_tokens = [
            self.processor.tokenizer.convert_tokens_to_ids(
                f"<|reserved_special_token_{pid+2}|>"
            )
            for pid in range(self.prompt_numbers)
        ]
        # (batch_size, num_tiles, channels, height, width)
        pixel_values = data["pixel_values"].to(self.model.device).to(self.model.dtype)
        mask_values = (
            torch.round((data["global_mask_values"] + 1.0) / 2.0 * 255.0)
            .long()
            .to(self.model.device)
        )
        mask_values = torch.clamp(mask_values, min=0, max=self.prompt_numbers)
        assert mask_values.max() < self.prompt_numbers + 1 and mask_values.min() >= 0

        mask_embeds = self.mask_patch_embedding(
            (mask_values != self.prompt_numbers).to(self.model.dtype)
        )  # binary mask
        input_ids = data["input_ids"]
        aspect_ratios = data["aspect_ratios"]
        bboxes = data["bboxes"]
        assert input_ids.shape[0] == 1, "Currently only support batch_size=1"

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        labels = data["labels"]

        image_features = None
        if pixel_values is not None:
            image_features = self.model.get_image_features(
                pixel_values=pixel_values,
                mask_embeds=mask_embeds,
            )
            image_features = image_features.to(
                inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            special_image_mask, _ = self.model.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        # feature replay
        new_inputs_embeds = []
        new_labels = []
        image_features_tiles = rearrange(
            image_features[1:].unsqueeze(0), "b n (h w) c -> b n c h w", h=16, w=16
        )
        for batch_idx in range(inputs_embeds.shape[0]):
            curr_inputs_embeds = inputs_embeds[batch_idx]
            curr_labels = labels[batch_idx]
            for crop_token in crop_tokens:
                if crop_token in input_ids[batch_idx]:
                    target_mask = input_ids[batch_idx].eq(crop_token)
                    target_indices = target_mask.nonzero().squeeze()
                    head_idx = target_indices.min().item()
                    tail_idx = target_indices.max().item()
                    image_features_recover = self._merge(
                        image_features_tiles,
                        aspect_ratios[batch_idx][0],
                        aspect_ratios[batch_idx][1],
                    )
                    feat_h, feat_w = image_features_recover.shape[2:]
                    x1, y1, x2, y2 = bboxes[batch_idx][str(crop_token)]
                    # RoI-Align
                    orig_h, orig_w = feat_h * 28, feat_w * 28  # 原图尺寸

                    # origin box
                    roi_orig_x1 = x1 * orig_w
                    roi_orig_y1 = y1 * orig_h
                    roi_orig_x2 = x2 * orig_w
                    roi_orig_y2 = y2 * orig_h

                    # feat box
                    spatial_scale = feat_w / orig_w
                    roi_feat_x1 = roi_orig_x1 * spatial_scale
                    roi_feat_y1 = roi_orig_y1 * spatial_scale
                    roi_feat_x2 = roi_orig_x2 * spatial_scale
                    roi_feat_y2 = roi_orig_y2 * spatial_scale

                    roi = torch.tensor(
                        [0, roi_feat_x1, roi_feat_y1, roi_feat_x2, roi_feat_y2],
                        dtype=torch.float32,
                        device=image_features_recover.device,
                    )

                    roi_features = torchvision.ops.roi_align(
                        input=image_features_recover.float(),
                        boxes=roi.unsqueeze(0),
                        output_size=(16, 16),
                        spatial_scale=spatial_scale,
                        sampling_ratio=2,
                        aligned=True,
                    )

                    image_features_replay = (
                        roi_features.permute(0, 2, 3, 1)
                        .flatten(1, 2)
                        .to(image_features_recover.dtype)
                        .squeeze()
                    )

                    curr_inputs_embeds = torch.cat(
                        [
                            curr_inputs_embeds[:head_idx],
                            image_features_replay,
                            curr_inputs_embeds[tail_idx + 1 :],
                        ]
                    )
                    curr_labels = torch.cat(
                        [
                            curr_labels[:head_idx],
                            -100
                            * torch.ones(
                                image_features_replay.shape[0],
                                dtype=torch.long,
                                device=labels.device,
                            ),
                            curr_labels[tail_idx + 1 :],
                        ]
                    )

                    assert (
                        curr_inputs_embeds.shape[0] == curr_labels.shape[0]
                    ), f"shape mismatch, got {curr_inputs_embeds.shape[0]} != {curr_labels.shape[0]}"

            new_inputs_embeds.append(curr_inputs_embeds.unsqueeze(0))
            new_labels.append(curr_labels)

        inputs_embeds = torch.cat(new_inputs_embeds, dim=0)
        labels = torch.cat(new_labels, dim=0)

        skip_this_batch = False

        if mode == "loss":
            position_ids = (
                torch.arange(
                    0,
                    inputs_embeds.shape[1],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
                .unsqueeze(0)
                .repeat(inputs_embeds.shape[0], 1)
            )
            attention_mask = torch.ones(
                inputs_embeds.shape[0],
                inputs_embeds.shape[1],
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            use_cache = False

            outputs, _skip_this_case = self._llm_forward(
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=use_cache,
            )

            if skip_this_batch or _skip_this_case:
                print("skip this batch!")
                loss_dict = {"loss": outputs.loss * 0.0}
            else:
                loss_dict = {"loss": outputs.loss}
            return loss_dict

        elif mode == "predict":
            pass
        elif mode == "tensor":
            pass
        else:
            raise NotImplementedError

    def _llm_forward(
        self,
        inputs_embeds: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict
            if return_dict is not None
            else self.model.config.use_return_dict
        )
        skip_this_case = False

        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs, skip_this_case
