# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import numpy as np
import torch
from mmengine.config import Config, ConfigDict
from mmengine.dist import master_only
from mmengine.fileio import PetrelBackend, get_file_backend
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, GenerationConfig
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto="auto"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert PTH model to HuggingFace model"
    )

    parser.add_argument(
        "--config",
        help="config file name or path.",
        default="./work_dirs/gar_8b/gar_8b.py",
    )
    parser.add_argument(
        "--pth_model",
        help="pth model file",
        default="./work_dirs/gar_8b/iter_37891.pth",
    )
    parser.add_argument(
        "--save_dir", help="the dir to save results", default="./work_dirs/gar_8b_hf"
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    torch.bfloat16

    torch.cuda.set_device(0)
    torch.distributed.init_process_group(backend="nccl")

    # build model
    if not os.path.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f"Cannot find {args.config}")

    # load config
    cfg = Config.fromfile(args.config)
    # if args.cfg_options is not None:
    # cfg.merge_from_dict(args.cfg_options)

    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load

    cfg.model.pretrained_pth = None
    selfmodel = BUILDER.build(cfg.model)

    backend = get_file_backend(args.pth_model)
    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio

        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)

    # load the state dict
    msg = selfmodel.load_state_dict(state_dict, strict=False)
    print(f"Load PTH model from {args.pth_model} with msg: {msg}")

    selfmodel.cuda()
    selfmodel.eval()
    selfmodel.to(torch.bfloat16)
    mllm_name_or_path = cfg.mllm_name_or_path

    processor = AutoProcessor.from_pretrained(
        mllm_name_or_path,
        trust_remote_code=True,
    )

    # convert to hf format
    from projects.grasp_any_region.hf_models.configuration_gar import GARConfig
    from projects.grasp_any_region.hf_models.modeling_gar import GARModel

    tokenizer = selfmodel.processor.tokenizer
    prompt_numbers = selfmodel.prompt_numbers
    crop_tokens_ids = [
        tokenizer.convert_tokens_to_ids(f"<|reserved_special_token_{pid+2}|>")
        for pid in range(prompt_numbers)
    ]

    base_config = AutoConfig.from_pretrained(
        mllm_name_or_path,
        trust_remote_code=True,
    )
    base_config.text_config.vocab_size = len(tokenizer)
    gar_config = GARConfig(
        mllm_config=base_config.to_dict(),
        prompt_numbers=prompt_numbers,
        crop_tokens_ids=crop_tokens_ids,
        auto_map={
            "AutoConfig": "configuration_gar.GARConfig",
            "AutoModel": "modeling_gar.GARModel",
            "AutoModelForCausalLM": "modeling_gar.GARModel",
        },
    )

    hf_model = GARModel(
        gar_config,
        mllm=selfmodel.model,
        mask_patch_embedding=selfmodel.mask_patch_embedding,
        use_flash_attn=True,
    )

    hf_model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
    selfmodel.processor.save_pretrained(args.save_dir)


if __name__ == "__main__":
    main()
