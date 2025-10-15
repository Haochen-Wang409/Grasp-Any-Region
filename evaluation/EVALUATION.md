# Evaluation of GAR

## 1. GARBench

### 1.1 GARBench-Caption-Simple

First, perform inference, e.g., using GAR-8B.

```bash
torchrun --nproc-per-node=1 --master-port=9811 \
    evaluation/GAR-Bench/inference.py \
    --model_name_or_path ByteDance/GAR-8B \
    --anno_file evaluation/GAR-Bench/annotations/GAR-Bench-Caption-Simple.json \
    --mode simple \
    --cache_name ${CACHE_NAME} \
    --data_type bf16 \
    --seed 42
```

The generated descriptions will be saved to ```evaluation/GAR-Bench/model_outputs/${CACHE_NAME}_simple.json```

Next, perform evaluation (with images using GPT-4o).

```bash
export AZURE_OPENAI_ENDPOINT=YOUR_AZURE_OPENAI_ENDPOINT
export AZURE_OPENAI_KEY=YOUR_AZURE_OPENAI_KEY

python3 evaluation/GAR-Bench/eval_simple.py --pred evaluation/GAR-Bench/model_outputs/${CACHE_NAME}_simple.json 
```

Reference cache (including model predictions and evaluation results) are stored in ```model_outputs/```. Due to the randomness during LLM-Judge, the final performance may slighly differ even with the same predicitons (even with ```temperature=0```).

To re-run the evaluation, you could change to your own ```CACHE_NAME```.

Reference results:

```bash
# GAR-1B
Accuracy:  0.5567010309278351

# GAR-8B
Accuracy:  0.6391752577319587
```

### 1.2 GARBench-Caption-Detailed

First, perform inference, e.g., using GAR-8B.

```bash
torchrun --nproc-per-node=1 --master-port=9811 \
    evaluation/GAR-Bench/inference.py \
    --model_name_or_path ByteDance/GAR-8B \
    --anno_file evaluation/GAR-Bench/annotations/GAR-Bench-Caption-Detailed.json \
    --mode detailed \
    --cache_name ${CACHE_NAME} \
    --data_type bf16 \
    --seed 42
```

The generated descriptions will be saved to ```evaluation/GAR-Bench/model_outputs/${CACHE_NAME}_detailed.json```

Next, perform evaluation (with images using GPT-4o).

```bash
export AZURE_OPENAI_ENDPOINT=YOUR_AZURE_OPENAI_ENDPOINT
export AZURE_OPENAI_KEY=YOUR_AZURE_OPENAI_KEY

python3 evaluation/GAR-Bench/eval_detailed.py --pred evaluation/GAR-Bench/model_outputs/${CACHE_NAME}_detailed.json 
```

Reference cache (including model predictions and evaluation results) are stored in ```model_outputs/```. Due to the randomness during LLM-Judge, the final performance may slighly differ even with the same predicitons (even with ```temperature=0```).

To re-run the evaluation, you could change to your own ```CACHE_NAME```.

Reference results:

```bash
# GAR-1B
Accuracy:  0.6635514018691588

# GAR-8B
Accuracy:  0.6915887850467289
```

### 1.3 GARBench-VQA

Perform inference, e.g., using GAR-8B.

```bash
torchrun --nproc-per-node=1 --master-port=9811 \
    evaluation/GAR-Bench/inference.py \
    --model_name_or_path ByteDance/GAR-8B \
    --anno_file evaluation/GAR-Bench/annotations/GAR-Bench-VQA.json \
    --mode vqa \
    --cache_name ${CACHE_NAME} \
    --data_type bf16 \
    --seed 42
```

Reference cache (including model predictions and evaluation results) are stored in ```model_outputs/```.

To re-run the evaluation, you could change to your own ```CACHE_NAME```.

Reference results:
```
# GAR-1B
color:           [34/69]=49.3
texture/pattern: [17/29]=58.6
mirror:          [36/61]=59.0
ordering:        [13/64]=20.3
material:        [14/36]=38.9
shape:           [32/64]=50.0
relation:        [57/101]=56.4
=> overall:      [203/424]=47.9

# GAR-8B
texture/pattern: [22/29]=75.9
material:        [19/36]=52.8
mirror:          [36/61]=59.0
relation:        [66/101]=65.4
shape:           [34/64]=53.1
ordering:        [28/64]=43.8
color:           [40/69]=58.0
=> overall:      [245/424]=57.8
```

## 2. DLC-Bench

First, download images of DLC-Bench and put the ```images``` folder in the ```annotations``` directory:
```bash
cd evaluation/DLC-Bench/annotations
hf download nvidia/DLC-Bench --repo-type dataset --include "images/*" --exclude "*" --local-dir ./
```

The overall structure should be:
```bash
evaluation/DLC-Bench/annotations
├── annotations.json              
├── class_names.json              
├── images
│   └── objects365_v2_*.jpg
└── qa.json
```

Next, perform inference to obtain detailed descriptions, e.g., using GAR-8B.

```bash
torchrun --nproc-per-node=1 --master-port=8841 \
    evaluation/DLC-Bench/inference.py \
    --model_name_or_path ByteDance/GAR-8B \
    --cache_name ${CACHE_NAME} \
    --data_type bf16 \
    --seed 42
```

The generated descriptions will be saved to ```evaluation/DLC-Bench/model_outputs/${CACHE_NAME}.json```

Finally, perform evaluation (with images using GPT-4o or without images using Llama3.1-8B).

**Optional 1. Using GPT-4o *with* images (Recommended)**

```bash
export AZURE_OPENAI_ENDPOINT=YOUR_AZURE_OPENAI_ENDPOINT
export AZURE_OPENAI_KEY=YOUR_AZURE_OPENAI_KEY

python3 evaluation/DLC-Bench/eval_gpt_with_image.py --pred evaluation/DLC-Bench/model_outputs/${CACHE_NAME}.json 
```

**Optional 2. Using Llama3.1-8B *without* images**

First, we need to serve Llama3.1-8B using vLLM.

```bash
bash evaluation/DLC-Bench/serve_judge.sh
```

Next, on the *same* node, run evaluation.

```bash
python3 eval_llama_without_image.py --pred ../model_outputs/${CACHE_NAME}.json --base_url http://localhost:8007/v1
```

For more details for the differences between these two evaluation settings, please refer to Appendix F of our paper.

Reference cache (including model predictions and evaluation results) are stored in ```model_outputs/```. Due to the randomness during LLM-Judge, the final performance may slighly differ even with the same predicitons (even with ```temperature=0```).

To re-run the evaluation, you could change to your own ```CACHE_NAME```.

Reference results:

```bash
# GAR-1B
# By GPT-4o (with images):
Summary (Pos    Neg     Avg(Pos, Neg)): 0.662,  0.880,  0.771
# By Llama3.1-8B (without images):
Summary (Pos    Neg     Avg(Pos, Neg)): 0.489,  0.870,  0.679

# GAR-8B
# By GPT-4o (with images):
Summary (Pos    Neg     Avg(Pos, Neg)): 0.680,  0.860,  0.770
# By Llama3.1-8B (without images):
Summary (Pos    Neg     Avg(Pos, Neg)): 0.502,  0.846,  0.674
```

## 3. Ferret-Bench

First, perform inference to obtain detailed descriptions, e.g., using GAR-8B.

```bash
torchrun --nproc-per-node=1 --master-port=8841 \
    evaluation/Ferret-Bench/inference.py \
    --model_name_or_path ByteDance/GAR-8B \
    --cache_name ${CACHE_NAME} \
    --data_type bf16 \
    --seed 42
```

The generated descriptions will be saved to ```evaluation/Ferret-Bench/model_outputs/${CACHE_NAME}.json```


Then, perform evaluation using GPT-4o.

```bash
export AZURE_OPENAI_ENDPOINT=YOUR_AZURE_OPENAI_ENDPOINT
export AZURE_OPENAI_KEY=YOUR_AZURE_OPENAI_KEY

cd evaluation/Ferret-Bench
bash eval.sh ${CACHE_NAME}
```

Reference model predictions are stored in ```model_outputs/```, and reference evaluation results are stored in ```gpt4_result/```. Due to the randomness during LLM-Judge, the final performance may slighly differ even with the same predicitons (even with ```temperature=0```).

To re-run the evaluation, you could change to your own ```CACHE_NAME```.

Reference results:
```bash
# GAR-1B
review_refer_desc
all 56.0
refer_desc 56.0
=================================

# GAR-8B
review_refer_desc
all 64.8
refer_desc 64.8
=================================
```


## 4. MDVP-Bench

First, perform inference to obtain detailed descriptions, e.g., using GAR-8B.

```bash
torchrun --nproc-per-node=1 --master-port=8841 \
    evaluation/MDVP-Bench/inference.py \
    --model_name_or_path ByteDance/GAR-8B \
    --cache_name ${CACHE_NAME} \
    --data_type bf16 \
    --seed 42
```

The generated descriptions will be saved to ```evaluation/MDVP-Bench/model_outputs/${CACHE_NAME}.json```


Then, perform evaluation using GPT-4o.

```bash
export AZURE_OPENAI_ENDPOINT=YOUR_AZURE_OPENAI_ENDPOINT
export AZURE_OPENAI_KEY=YOUR_AZURE_OPENAI_KEY

cd evaluation/MDVP-Bench
bash eval.sh ${CACHE_NAME}
```

Reference model predictions are stored in ```model_outputs/```. Due to the randomness during LLM-Judge, the final performance may slighly differ even with the same predicitons (even with ```temperature=0```).

To re-run the evaluation, you could change to your own ```CACHE_NAME```.

Reference results:
```bash
# GAR-1B
android_detailed_caption_box 80.65
multipanel_detailed_caption_box 103.7
natural_detailed_caption_box 152.63
ocr_doc_detailed_caption_box 146.87
ocr_spotting_detailed_caption_box 152.38
web_detailed_caption_box 150.0
# Natural = natural_detailed_caption_box = 152.6
# OCR = (ocr_doc_detailed_caption_box + ocr_spotting_detailed_caption_box) / 2 = 149.6
# Multi-Panel = multipanel_detailed_caption_box = 103.7
# Sceenshot = (android_detailed_caption_box + web_detailed_caption_box) / 2 = 115.3

# GAR-8B
android_detailed_caption_box 113.79
multipanel_detailed_caption_box 117.24
natural_detailed_caption_box 178.57
ocr_doc_detailed_caption_box 138.10
ocr_spotting_detailed_caption_box 160.0
web_detailed_caption_box 132.26
# Natural = natural_detailed_caption_box = 178.6
# OCR = (ocr_doc_detailed_caption_box + ocr_spotting_detailed_caption_box) / 2 = 149.1
# Multi-Panel = multipanel_detailed_caption_box = 117.2
# Sceenshot = (android_detailed_caption_box + web_detailed_caption_box) / 2 = 123.0
```