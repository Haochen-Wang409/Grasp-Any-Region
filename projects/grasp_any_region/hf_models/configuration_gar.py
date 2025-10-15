import copy

from transformers import AutoConfig, PerceptionLMConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class GARConfig(PretrainedConfig):
    model_type = "GAR"
    is_composition = True

    def __init__(
        self,
        mllm_config=None,
        prompt_numbers=5,
        crop_tokens_ids=[128004, 128005, 128008, 128010, 128011],
        use_flash_attn=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if mllm_config is None:
            mllm_config = {}
            logger.info(
                "mllm_config is None. Initializing the PerceptionLM with default values."
            )

        if mllm_config is None:
            self.mllm_config = AutoConfig.from_pretrained("facebook/Perception-LM-1B")
        else:
            self.mllm_config = PerceptionLMConfig(**mllm_config)
        self.prompt_numbers = prompt_numbers

        self.crop_tokens_ids = crop_tokens_ids
        assert (
            len(self.crop_tokens_ids) == self.prompt_numbers
        ), f"{self.crop_tokens_ids} crop_tokens_ids length should be {self.prompt_numbers}"

        try:
            self.patch_size_h = (
                self.mllm_config.vision_config.model_args["img_size"][0]
                // self.mllm_config.vision_config.model_args["ref_feat_shape"][0]
            )
            self.patch_size_w = (
                self.mllm_config.vision_config.model_args["img_size"][1]
                // self.mllm_config.vision_config.model_args["ref_feat_shape"][1]
            )
            self.kernel_size = [self.patch_size_h, self.patch_size_w]
        except:
            self.patch_size_h = 16
            self.patch_size_w = 16
            self.kernel_size = [self.patch_size_h, self.patch_size_w]

        try:
            self.mask_path_embedding_out_channels = (
                self.mllm_config.vision_config.num_features
            )
        except:
            self.mask_path_embedding_out_channels = 1280

        self.mllm_config.use_flash_attn = True if use_flash_attn else False
        self.mllm_config.text_config.use_flash_attn = True if use_flash_attn else False
        self.mllm_config.vision_config.use_flash_attn = False

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["mllm_config"] = self.mllm_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
