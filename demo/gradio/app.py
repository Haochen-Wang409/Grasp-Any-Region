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
import base64
import io

import cv2
import gradio as gr
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from transformers import (
    AutoModel,
    AutoProcessor,
    GenerationConfig,
    SamModel,
    SamProcessor,
)

try:
    from spaces import GPU
except ImportError:
    print("Spaces not installed, using dummy GPU decorator")

    def GPU(*args, **kwargs):
        def decorator(fn):
            return fn

        return decorator


from evaluation.eval_dataset import SingleRegionCaptionDataset

# Load SAM model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

# Initialize the captioning model and processor
model_path = "Haochen-Wang409/GAR-1B"
model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
).eval()

processor = AutoProcessor.from_pretrained(
    model_path,
    trust_remote_code=True,
)


@GPU(duration=75)
def image_to_sam_embedding(base64_image):
    try:
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(base64_image)

        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Process image with SAM processor
        inputs = sam_processor(image, return_tensors="pt").to(device)

        # Get image embedding
        with torch.no_grad():
            image_embedding = sam_model.get_image_embeddings(inputs["pixel_values"])

        # Convert to CPU and numpy
        image_embedding = image_embedding.cpu().numpy()

        # Encode the embedding as base64
        embedding_bytes = image_embedding.tobytes()
        embedding_base64 = base64.b64encode(embedding_bytes).decode("utf-8")

        return embedding_base64
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise gr.Error(f"Failed to process image: {str(e)}")


@GPU(duration=75)
def describe(image_base64: str, mask_base64: str, query: str):
    # Convert base64 to PIL Image
    image_bytes = base64.b64decode(
        image_base64.split(",")[1] if "," in image_base64 else image_base64
    )
    img = Image.open(io.BytesIO(image_bytes))
    mask_bytes = base64.b64decode(
        mask_base64.split(",")[1] if "," in mask_base64 else mask_base64
    )
    mask = Image.open(io.BytesIO(mask_bytes))
    mask = np.array(mask.convert("L"))

    prompt_number = model.config.prompt_numbers
    prompt_tokens = [f"<Prompt{i_p}>" for i_p in range(prompt_number)] + ["<NO_Prompt>"]

    # Assuming mask is given as a numpy array and the image is a PIL image
    dataset = SingleRegionCaptionDataset(
        image=img,
        mask=mask,
        processor=processor,
        prompt_number=prompt_number,
        visual_prompt_tokens=prompt_tokens,
        data_dtype=torch.bfloat16,
    )

    data_sample = dataset[0]

    # Generate the caption
    with torch.no_grad():
        generate_ids = model.generate(
            **data_sample,
            generation_config=GenerationConfig(
                max_new_tokens=1024,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            ),
            return_dict=True,
        )

    output_caption = processor.tokenizer.decode(
        generate_ids.sequences[0], skip_special_tokens=True
    ).strip()

    # Stream the tokens
    text = ""
    for token in output_caption:
        text += token
        yield text


@GPU(duration=75)
def describe_without_streaming(image_base64: str, mask_base64: str, query: str):
    # Convert base64 to PIL Image
    image_bytes = base64.b64decode(
        image_base64.split(",")[1] if "," in image_base64 else image_base64
    )
    img = Image.open(io.BytesIO(image_bytes))
    mask_bytes = base64.b64decode(
        mask_base64.split(",")[1] if "," in mask_base64 else mask_base64
    )
    mask = Image.open(io.BytesIO(mask_bytes))
    mask = np.array(mask.convert("L"))
    prompt_number = model.config.prompt_numbers
    prompt_tokens = [f"<Prompt{i_p}>" for i_p in range(prompt_number)] + ["<NO_Prompt>"]

    # Assuming mask is given as a numpy array and the image is a PIL image
    dataset = SingleRegionCaptionDataset(
        image=img,
        mask=mask,
        processor=processor,
        prompt_number=prompt_number,
        visual_prompt_tokens=prompt_tokens,
        data_dtype=torch.bfloat16,
    )

    data_sample = dataset[0]

    # Generate the caption
    with torch.no_grad():
        generate_ids = model.generate(
            **data_sample,
            generation_config=GenerationConfig(
                max_new_tokens=1024,
                # do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id,
                pad_token_id=processor.tokenizer.pad_token_id,
            ),
            return_dict=True,
        )

    output_caption = processor.tokenizer.decode(
        generate_ids.sequences[0], skip_special_tokens=True
    ).strip()

    return output_caption


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Describe Anything gradio demo")
    parser.add_argument(
        "--server_addr",
        "--host",
        type=str,
        default=None,
        help="The server address to listen on.",
    )
    parser.add_argument(
        "--server_port", "--port", type=int, default=None, help="The port to listen on."
    )

    args = parser.parse_args()

    # Create Gradio interface
    with gr.Blocks() as demo:
        gr.Interface(
            fn=image_to_sam_embedding,
            inputs=gr.Textbox(label="Image Base64"),
            outputs=gr.Textbox(label="Embedding Base64"),
            title="Image Embedding Generator",
            api_name="image_to_sam_embedding",
        )
        gr.Interface(
            fn=describe,
            inputs=[
                gr.Textbox(label="Image Base64"),
                gr.Text(label="Mask Base64"),
                gr.Text(label="Prompt"),
            ],
            outputs=[gr.Text(label="Description")],
            title="Mask Description Generator",
            api_name="describe",
        )
        gr.Interface(
            fn=describe_without_streaming,
            inputs=[
                gr.Textbox(label="Image Base64"),
                gr.Text(label="Mask Base64"),
                gr.Text(label="Prompt"),
            ],
            outputs=[gr.Text(label="Description")],
            title="Mask Description Generator (Non-Streaming)",
            api_name="describe_without_streaming",
        )

    demo._block_thread = demo.block_thread
    demo.block_thread = lambda: None
    demo.launch(
        share=True,
        server_name=args.server_addr,
        server_port=args.server_port,
        ssr_mode=False,
    )

    for route in demo.app.routes:
        if route.path == "/":
            demo.app.routes.remove(route)
    demo.app.mount("/", StaticFiles(directory="dist", html=True), name="demo")

    demo._block_thread()
