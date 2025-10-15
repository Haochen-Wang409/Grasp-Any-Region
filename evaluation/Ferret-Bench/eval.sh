CHECKPOINT_FILE=$1

mkdir -p gpt4_result/${CHECKPOINT_FILE}
mkdir -p gpt4_result/${CHECKPOINT_FILE}/refer_desc

python3 eval_gpt.py \
    --question ferret_gpt4_data/refer_desc/question.jsonl \
    --context ferret_gpt4_data/refer_desc/context.jsonl \
    --answer-list \
    ferret_gpt4_data/refer_desc/answer.jsonl \
    gpt4_result/${CHECKPOINT_FILE}/refer_desc/ferret_answer.jsonl \
    --rule ferret_gpt4_data/rule.json \
    --output gpt4_result/${CHECKPOINT_FILE}/review_refer_desc.jsonl \
    --source-file model_outputs/${CHECKPOINT_FILE}.json

python3 summarize_gpt_review.py  \
    --dir=gpt4_result/${CHECKPOINT_FILE}