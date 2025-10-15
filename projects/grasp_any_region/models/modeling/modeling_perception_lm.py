# *************************************************************************
# This file may have been modified by Bytedance Inc. (“Bytedance Inc.'s Mo-
# difications”). All Bytedance Inc.'s Modifications are Copyright (2025) B-
# ytedance Inc..
# *************************************************************************

# Adapted from https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/perception_lm/modeling_perception_lm.py

# coding=utf-8
# Copyright 2025 Meta Platforms, Inc. and the HuggingFace Inc. team. All rights reserved.
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

import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from timm.models._manipulate import checkpoint
from torch import nn
from transformers import AutoModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import auto_docstring, can_return_tuple

from .configuration_perception_lm import PerceptionLMConfig


class PerceptionLMAdaptiveAvgPooling(nn.Module):
    def __init__(self, pooling_ratio=2):
        super().__init__()
        self.pooling_ratio = pooling_ratio

    def forward(self, hidden_states):
        b, num_tokens, c = hidden_states.shape
        h = int(math.sqrt(num_tokens))
        if h * h != num_tokens:
            raise ValueError(
                f"num_tokens {num_tokens} is expected to be a square number"
            )

        shape = (h // self.pooling_ratio, h // self.pooling_ratio)
        hidden_states = hidden_states.permute(0, 2, 1).reshape(b, -1, h, h)
        hidden_states = F.adaptive_avg_pool2d(hidden_states, shape)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        return hidden_states


class PerceptionLMMultiModalProjector(nn.Module):
    def __init__(self, config: PerceptionLMConfig):
        super().__init__()
        input_size = config.vision_config.model_args["embed_dim"]
        output_size = config.text_config.hidden_size
        self.linear_1 = nn.Linear(
            in_features=input_size,
            out_features=output_size,
            bias=True,
        )
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(
            in_features=output_size,
            out_features=output_size,
            bias=True,
        )
        self.pooling = (
            PerceptionLMAdaptiveAvgPooling(config.projector_pooling_ratio)
            if config.projector_pooling_ratio > 1
            else nn.Identity()
        )

    def forward(self, features):
        features = features.permute(1, 0, 2)  # NLD -> LND
        features = self.linear_1(features)
        features = self.gelu(features)
        features = self.linear_2(features)
        features = features.permute(1, 0, 2)  # LND -> NLD
        features = self.pooling(features)
        return features


@auto_docstring
class PerceptionLMPreTrainedModel(PreTrainedModel):
    config: PerceptionLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"

    _supports_flash_attn = True
    _supports_sdpa = True

    _can_compile_fullgraph = True
    _supports_flex_attn = True
    _supports_attention_backend = True


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for PerceptionLM outputs, with hidden states and attentions.
    """
)
class PerceptionLMModelOutputWithPast(BaseModelOutputWithPast):
    r"""
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        Image hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    video_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_videos, sequence_length, hidden_size)`.
        Video hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    image_hidden_states: Optional[torch.FloatTensor] = None

    video_hidden_states: Optional[torch.FloatTensor] = None


@dataclass
@auto_docstring(
    custom_intro="""
    Base class for PerceptionLM causal language model (or autoregressive) outputs.
    """
)
class PerceptionLMCausalLMOutputWithPast(ModelOutput):
    r"""
    loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
        Language modeling loss (for next-token prediction).
    logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
        Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
    past_key_values (`Cache`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
        Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
        `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

        Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
        `past_key_values` input) to speed up sequential decoding.
    image_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_images, sequence_length, hidden_size)`.
        Image hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    video_hidden_states (`torch.FloatTensor`, *optional*):
        A `torch.FloatTensor` of size `(batch_size, num_videos, sequence_length, hidden_size)`.
        Video hidden_states of the model produced by the vision encoder and after projecting the last hidden state.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[list[torch.FloatTensor]] = None
    hidden_states: Optional[tuple[torch.FloatTensor]] = None
    attentions: Optional[tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[torch.FloatTensor] = None

    video_hidden_states: Optional[torch.FloatTensor] = None


@auto_docstring
class PerceptionLMModel(PerceptionLMPreTrainedModel):
    _checkpoint_conversion_mapping = {}

    def __init__(self, config: PerceptionLMConfig):
        super().__init__(config)
        self.vision_tower = AutoModel.from_config(config.vision_config)

        def custom_forward_features(
            self,
            x: torch.Tensor,
            mask_embeds: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """Forward pass through feature extraction layers.

            Args:
                x: Input tensor.

            Returns:
                Feature tensor.
            """
            x = self.patch_embed(x)
            if mask_embeds is not None:
                x = x + mask_embeds.flatten(2).transpose(1, 2)
            x, rot_pos_embed = self._pos_embed(x)
            x = self.norm_pre(x)

            if getattr(self, "rope_mixed", False) and rot_pos_embed is not None:
                # Handle depth-dependent embeddings for mixed mode
                # pos embed has shape (depth, num_heads, H*W, dim) or (depth, batch_size, num_heads, H*W, dim)
                for i, blk in enumerate(self.blocks):
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        x = checkpoint(blk, x, rope=rot_pos_embed[i])
                    else:
                        x = blk(x, rope=rot_pos_embed[i])
            else:
                # Standard path for non-mixed mode
                for blk in self.blocks:
                    if self.grad_checkpointing and not torch.jit.is_scripting():
                        x = checkpoint(blk, x, rope=rot_pos_embed)
                    else:
                        x = blk(x, rope=rot_pos_embed)

            x = self.norm(x)
            return x

        self.vision_tower.timm_model.forward_features = custom_forward_features.__get__(
            self.vision_tower.timm_model
        )

        self.multi_modal_projector = PerceptionLMMultiModalProjector(config)
        self.language_model = AutoModel.from_config(config.text_config)
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        mask_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, num_tiles, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(num_tiles, num_patches, embed_dim)`).
        """
        if len(pixel_values.shape) == 5:
            pixel_values = pixel_values.flatten(0, 1)
        assert (
            len(pixel_values.shape) == 4
        ), f"pixel_values should be of shape (batch_size * num_tiles, channels, height, width). But got {pixel_values.shape}."
        # pre-mask
        image_outputs = self.vision_tower(pixel_values, mask_embeds=mask_embeds)
        # image_outputs = self.vision_tower(pixel_values)
        image_outputs = image_outputs.last_hidden_state
        if self.config.vision_use_cls_token:
            image_outputs = image_outputs[:, 1:, :]
        # post-mask
        # if mask_embeds is not None:
        #     image_outputs = image_outputs + mask_embeds.flatten(2).transpose(1, 2)
        image_features = self.multi_modal_projector(image_outputs)
        return image_features

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor = None,
        video_features: torch.FloatTensor = None,
    ):
        """
        Obtains multimodal placeholdr mask from `input_ids` or `inputs_embeds`, and checks that the placeholder token count is
        equal to the length of multimodal features. If the lengths are different, an error is raised.
        """
        if input_ids is None:
            special_image_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(
                    self.config.image_token_id,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )
            special_image_mask = special_image_mask.all(-1)
            special_video_mask = inputs_embeds == self.get_input_embeddings()(
                torch.tensor(
                    self.config.video_token_id,
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
            )
            special_video_mask = special_video_mask.all(-1)
        else:
            special_image_mask = input_ids == self.config.image_token_id
            special_video_mask = input_ids == self.config.video_token_id

        n_image_tokens = special_image_mask.sum()
        special_image_mask = (
            special_image_mask.unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        if (
            image_features is not None
            and inputs_embeds[special_image_mask].numel() != image_features.numel()
        ):
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {image_features.size()[:-1].numel()}"
            )

        n_video_tokens = special_video_mask.sum()
        special_video_mask = (
            special_video_mask.unsqueeze(-1)
            .expand_as(inputs_embeds)
            .to(inputs_embeds.device)
        )
        if (
            video_features is not None
            and inputs_embeds[special_video_mask].numel() != video_features.numel()
        ):
            raise ValueError(
                f"Videos features and image tokens do not match: tokens: {n_video_tokens}, features {video_features.size()[:-1].numel()}"
            )

        return special_image_mask, special_video_mask

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        mask_embeds: Optional[torch.FloatTensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # need
        position_ids: Optional[torch.LongTensor] = None,  # need
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,  # need
        use_cache: Optional[bool] = None,  # need
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Union[tuple, PerceptionLMModelOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if (
            pixel_values is not None or pixel_values_videos is not None
        ) and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both (pixel_values or pixel_values_videos) and inputs_embeds at the same time, and must specify either one"
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values=pixel_values, mask_embeds=mask_embeds
            )
            image_features = image_features.to(
                inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            special_image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_image_mask, image_features
            )

        video_features = None
        if pixel_values_videos is not None:
            video_features = self.get_image_features(pixel_values=pixel_values_videos)
            video_features = video_features.to(
                inputs_embeds.device, dtype=inputs_embeds.dtype
            )
            _, special_video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_features
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                special_video_mask, video_features
            )

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )
        return PerceptionLMModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
            past_key_values=outputs.past_key_values,
            attentions=outputs.attentions,
            image_hidden_states=image_features if pixel_values is not None else None,
            video_hidden_states=(
                video_features if pixel_values_videos is not None else None
            ),
        )


@auto_docstring
class PerceptionLMForConditionalGeneration(
    PerceptionLMPreTrainedModel, GenerationMixin
):
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: PerceptionLMConfig):
        super().__init__(config)
        self.model = PerceptionLMModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )
        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_decoder(self, decoder):
        self.model.set_decoder(decoder)

    def get_decoder(self):
        return self.model.get_decoder()

    def get_image_features(
        self,
        pixel_values: torch.FloatTensor,
        mask_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        return self.model.get_image_features(
            pixel_values=pixel_values, mask_embeds=mask_embeds, **kwargs
        )

    def get_placeholder_mask(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.FloatTensor,
        image_features: torch.FloatTensor = None,
        video_features: torch.FloatTensor = None,
    ):
        return self.model.get_placeholder_mask(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_features,
            video_features=video_features,
        )

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # no need
        pixel_values: Optional[torch.FloatTensor] = None,  # no need
        pixel_values_videos: Optional[torch.FloatTensor] = None,  # no need
        attention_mask: Optional[torch.Tensor] = None,  # need
        position_ids: Optional[torch.LongTensor] = None,  # need
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,  # need
        labels: Optional[torch.LongTensor] = None,  # need
        use_cache: Optional[bool] = None,  # need
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **lm_kwargs,
    ) -> Union[tuple, PerceptionLMCausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, PerceptionLMForConditionalGeneration

        >>> model = PerceptionLMForConditionalGeneration.from_pretrained("perception_lm-hf/perception_lm-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("perception_lm-hf/perception_lm-1.5-7b-hf")

        >>> prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(images=image, text=prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_new_tokens=15)
        >>> processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "USER:  \nWhat's the content of the image? ASSISTANT: The image features a busy city street with a stop sign prominently displayed"
        ```"""
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None

        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.text_config.vocab_size,
                **lm_kwargs,
            )

        return PerceptionLMCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            video_hidden_states=outputs.video_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        mask_embeds=None,
        pixel_values_videos=None,
        attention_mask=None,
        cache_position=None,
        logits_to_keep=None,
        feature_replay=None,
        feature_replay_video=None,
        crop_tokens=[128004],
        roi_align=None,
        bboxes=None,
        aspect_ratios=True,
        processor=None,
        **kwargs,
    ):
        # Overwritten -- in specific circumstances we don't want to forward image inputs to the model

        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        assert not (feature_replay and feature_replay_video)

        if cache_position[0] == 0:
            inputs_embeds = model_inputs["inputs_embeds"]

            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)

            image_features = None
            if pixel_values is not None:
                image_features = self.get_image_features(
                    pixel_values=pixel_values, mask_embeds=mask_embeds
                )
                image_features = image_features.to(
                    inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                special_image_mask, _ = self.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    image_features=image_features,
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    special_image_mask, image_features
                )

            video_features = None
            if pixel_values_videos is not None:
                video_features = self.get_image_features(
                    pixel_values=pixel_values_videos
                )
                video_features = video_features.to(
                    inputs_embeds.device, dtype=inputs_embeds.dtype
                )
                _, special_video_mask = self.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=inputs_embeds,
                    video_features=video_features,
                )
                inputs_embeds = inputs_embeds.masked_scatter(
                    special_video_mask, video_features
                )

            if feature_replay:
                assert (
                    inputs_embeds.shape[0] == 1
                ), "Currently only support batch_size=1 for feature replay"

                def _merge(tiles: torch.Tensor, ncw: int, nch: int) -> torch.Tensor:
                    # merge image tiles to the original image
                    # input: (batch_size, ncw * nch, num_channels, height//nch, width//ncw)
                    # output: (batch_size, num_channels, height, width)

                    batch_size, num_tiles, num_channels, tile_height, tile_width = (
                        tiles.size()
                    )
                    assert num_tiles == ncw * nch, f"{ncw * nch} != {num_tiles}"

                    tiles = tiles.view(
                        batch_size, nch, ncw, num_channels, tile_height, tile_width
                    )
                    tiles = tiles.permute(0, 3, 1, 4, 2, 5).contiguous()

                    original_height = nch * tile_height
                    original_width = ncw * tile_width

                    image = tiles.view(
                        batch_size, num_channels, original_height, original_width
                    )

                    return image

                new_inputs_embeds = []
                image_features_tiles = rearrange(
                    image_features[1:].unsqueeze(0),
                    "b n (h w) c -> b n c h w",
                    h=16,
                    w=16,
                )
                for batch_idx in range(inputs_embeds.shape[0]):
                    curr_inputs_emebds = inputs_embeds[batch_idx]
                    for crop_token in crop_tokens:
                        if crop_token in input_ids[batch_idx]:
                            target_mask = input_ids[batch_idx].eq(crop_token)
                            target_indices = target_mask.nonzero().squeeze()
                            head_idx = target_indices.min().item()
                            tail_idx = target_indices.max().item()
                            image_features_recover = _merge(
                                image_features_tiles,
                                aspect_ratios[batch_idx][0],
                                aspect_ratios[batch_idx][1],
                            )
                            x1, y1, x2, y2 = bboxes[batch_idx][str(crop_token)]
                            feat_h, feat_w = image_features_recover.shape[2:]
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

                            curr_inputs_emebds = torch.cat(
                                [
                                    inputs_embeds[batch_idx][:head_idx],
                                    image_features_replay,
                                    inputs_embeds[batch_idx][tail_idx + 1 :],
                                ]
                            )

                    new_inputs_embeds.append(curr_inputs_emebds.unsqueeze(0))

                inputs_embeds = torch.cat(new_inputs_embeds, dim=0)
                model_inputs["position_ids"] = (
                    torch.arange(
                        0,
                        inputs_embeds.shape[1],
                        dtype=torch.long,
                        device=inputs_embeds.device,
                    )
                    .unsqueeze(0)
                    .repeat(inputs_embeds.shape[0], 1)
                )
                model_inputs["attention_mask"] = torch.ones(
                    inputs_embeds.shape[0],
                    inputs_embeds.shape[1],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
                model_inputs["cache_position"] = model_inputs["position_ids"].clone()

            elif feature_replay_video:
                assert (
                    inputs_embeds.shape[0] == 1
                ), "Currently only support batch_size=1 for feature replay"
                assert processor is not None, "Need processor"

                new_inputs_embeds = []
                image_features_tiles = rearrange(
                    image_features.unsqueeze(0), "b n (h w) c -> b n c h w", h=16, w=16
                )
                for batch_idx in range(inputs_embeds.shape[0]):
                    curr_inputs_emebds = inputs_embeds[batch_idx]
                    for frame_idx in range(image_features.shape[0]):
                        crop_token = processor.tokenizer.convert_tokens_to_ids(
                            f"<|reserved_special_token_{2 + frame_idx}|>"
                        )
                        if crop_token in input_ids[batch_idx]:
                            target_mask = input_ids[batch_idx].eq(crop_token)
                            target_indices = target_mask.nonzero().squeeze()
                            head_idx = target_indices.min().item()
                            tail_idx = target_indices.max().item()
                            x1, y1, x2, y2 = bboxes[batch_idx][str(crop_token)]
                            feat_h, feat_w = 16, 16
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
                                device=image_features_tiles.device,
                            )

                            roi_features = torchvision.ops.roi_align(
                                input=image_features_tiles[:, frame_idx].float(),
                                boxes=roi.unsqueeze(0),
                                output_size=(16, 16),
                                spatial_scale=spatial_scale,
                                sampling_ratio=2,
                                aligned=True,
                            )

                            image_features_replay = (
                                roi_features.permute(0, 2, 3, 1)
                                .flatten(1, 2)
                                .to(image_features_tiles.dtype)
                                .squeeze()
                            )

                            curr_inputs_emebds = torch.cat(
                                [
                                    curr_inputs_emebds[:head_idx],
                                    image_features_replay,
                                    curr_inputs_emebds[tail_idx + 1 :],
                                ]
                            )

                    new_inputs_embeds.append(curr_inputs_emebds.unsqueeze(0))

                inputs_embeds = torch.cat(new_inputs_embeds, dim=0)
                model_inputs["position_ids"] = (
                    torch.arange(
                        0,
                        inputs_embeds.shape[1],
                        dtype=torch.long,
                        device=inputs_embeds.device,
                    )
                    .unsqueeze(0)
                    .repeat(inputs_embeds.shape[0], 1)
                )
                model_inputs["attention_mask"] = torch.ones(
                    inputs_embeds.shape[0],
                    inputs_embeds.shape[1],
                    dtype=torch.long,
                    device=inputs_embeds.device,
                )
                model_inputs["cache_position"] = model_inputs["position_ids"].clone()

            model_inputs["inputs_embeds"] = inputs_embeds
            model_inputs["input_ids"] = None
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["mask_embeds"] = None

        return model_inputs


__all__ = [
    "PerceptionLMForConditionalGeneration",
    "PerceptionLMPreTrainedModel",
    "PerceptionLMModel",
]
