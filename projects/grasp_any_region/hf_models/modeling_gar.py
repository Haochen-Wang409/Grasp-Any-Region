from typing import List, Optional, Tuple, Union

import torch
import torchvision
from einops import rearrange
from torch import nn
from transformers import GenerationConfig, PerceptionLMForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_gar import GARConfig

logger = logging.get_logger(__name__)


class GARModel(PreTrainedModel):
    config_class = GARConfig
    main_input_name = "pixel_values"
    base_model_prefix = "language_model"
    _no_split_modules = ["LlamaDecoderLayer"]
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: GARConfig,
        mllm=None,
        mask_patch_embedding=None,
        use_flash_attn=True,
    ):
        super().__init__(config)
        use_flash_attn = use_flash_attn
        config.mllm_config.use_flash_attn = True if use_flash_attn else False
        config.mllm_config.text_config.use_flash_attn = (
            True if use_flash_attn else False
        )
        config.mllm_config.vision_config.use_flash_attn = False

        config.mllm_config._attn_implementation = (
            "flash_attention_2" if use_flash_attn else "eager"
        )
        config.mllm_config.vision_config._attn_implementation = "eager"

        self.prompt_numbers = config.prompt_numbers

        if mllm is not None:
            self.mllm = mllm
        else:
            self.mllm = PerceptionLMForConditionalGeneration(config.mllm_config)
        if mask_patch_embedding is not None:
            self.mask_patch_embedding = mask_patch_embedding
        else:
            self.mask_patch_embedding = nn.Conv2d(
                in_channels=3,
                out_channels=config.mask_path_embedding_out_channels,
                kernel_size=config.kernel_size,
                stride=config.kernel_size,
                bias=False,
            )

        self.crop_tokens_ids = config.crop_tokens_ids

    @property
    def lm_head(self):
        return self.mllm.model.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.mllm.model.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.mllm.model.language_model.get_output_embeddings()

    def forward(self, data, data_samples=None, mode="loss"):
        crop_tokens = self.crop_tokens_ids
        # (batch_size, num_tiles, channels, height, width)
        pixel_values = data["pixel_values"].to(self.mllm.device).to(self.mllm.dtype)
        mask_values = (
            torch.round((data["global_mask_values"] + 1.0) / 2.0 * 255.0)
            .long()
            .to(self.mllm.device)
        )
        mask_values = torch.clamp(mask_values, min=0, max=self.prompt_numbers)
        assert mask_values.max() < self.prompt_numbers + 1 and mask_values.min() >= 0

        mask_embeds = self.mask_patch_embedding(
            (mask_values != self.prompt_numbers).to(self.mllm.dtype)
        )  # binary mask
        input_ids = data["input_ids"]
        aspect_ratios = data["aspect_ratios"]
        bboxes = data["bboxes"]
        assert input_ids.shape[0] == 1, "Currently only support batch_size=1"

        inputs_embeds = self.mllm.get_input_embeddings()(input_ids)
        labels = data["labels"]

        image_features = None
        if pixel_values is not None:
            image_features = self.mllm.get_image_features(
                pixel_values=pixel_values,
                mask_embeds=mask_embeds,
            )
            image_features = image_features.to(
                inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            special_image_mask, _ = self.mllm.get_placeholder_mask(
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
                    orig_h, orig_w = feat_h * 28, feat_w * 28

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

        return outputs

    def _merge(self, tiles: torch.Tensor, ncw: int, nch: int) -> torch.Tensor:
        batch_size, num_tiles, num_channels, tile_height, tile_width = tiles.size()
        assert num_tiles == ncw * nch, f"{ncw * nch} != {num_tiles}"

        tiles = tiles.view(batch_size, nch, ncw, num_channels, tile_height, tile_width)
        tiles = tiles.permute(0, 3, 1, 4, 2, 5).contiguous()

        original_height = nch * tile_height
        original_width = ncw * tile_width

        image = tiles.view(batch_size, num_channels, original_height, original_width)

        return image

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
            return_dict if return_dict is not None else self.mllm.config.use_return_dict
        )
        skip_this_case = False

        outputs = self.mllm(
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

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        global_mask_values: Optional[torch.LongTensor] = None,
        aspect_ratios: Optional[torch.FloatTensor] = None,
        bboxes: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        device = self.device

        if pixel_values is not None:
            pixel_values = pixel_values.to(device).to(self.mllm.dtype)
            if global_mask_values is not None:

                mask_values = (
                    torch.round((global_mask_values + 1.0) / 2.0 * 255.0)
                    .long()
                    .to(device)
                )
                mask_values = torch.clamp(mask_values, min=0, max=self.prompt_numbers)

                assert (
                    mask_values.max() < self.prompt_numbers + 1
                    and mask_values.min() >= 0
                ), f"max: {mask_values.max()}, min: {mask_values.min()}"
                mask_embeds = self.mask_patch_embedding(
                    (mask_values != self.prompt_numbers).to(self.mllm.dtype)
                )
            else:
                mask_embeds = None

            inputs_embeds = self.mllm.get_input_embeddings()(input_ids)

            image_features = self.mllm.get_image_features(
                pixel_values=pixel_values,
                mask_embeds=mask_embeds,
            )
            image_features = image_features.to(
                inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            special_image_mask, _ = self.mllm.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

            # feature replay
            new_inputs_embeds = []
            image_features_tiles = rearrange(
                image_features[1:].unsqueeze(0), "b n (h w) c -> b n c h w", h=16, w=16
            )
            for batch_idx in range(inputs_embeds.shape[0]):
                curr_inputs_embeds = inputs_embeds[batch_idx]
                for crop_token in self.crop_tokens_ids:
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
                        orig_h, orig_w = feat_h * 28, feat_w * 28

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

                new_inputs_embeds.append(curr_inputs_embeds.unsqueeze(0))
            inputs_embeds = torch.cat(new_inputs_embeds, dim=0)
        else:
            inputs_embeds = self.mllm.get_input_embeddings()(input_ids)

        outputs = self.mllm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
            use_cache=True,
            return_dict_in_generate=True,
        )

        return outputs
