export output_results=$1

python3 transfer.py --output_path $output_results

for p in \
    "android_detailed_caption_box" \
    "multipanel_detailed_caption_box" \
    "natural_detailed_caption_box" \
    "ocr_doc_detailed_caption_box" \
    "ocr_spotting_detailed_caption_box" \
    "web_detailed_caption_box"
do
    python3 eval_gpt.py --phase $p
    python3 summarize_gpt_score.py --dir result
    rm -fr result/*
done
