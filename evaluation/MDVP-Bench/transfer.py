import argparse
import json
import os

"""
question:
{
    "question_id": 1, 
    "image": "000000104486.jpg", 
    "category": "natural_box", 
    "text": "Please analyze the relationship between all marked regions in the image.", 
    "annotation": {
        "bbox": [[157.23, 341.07, 10.67, 2.08]], 
        "segmentation": []
    }
}
answer:
{
    "question_id": 1, 
    "image": "000000104486.jpg", 
    "category": "natural_box", 
    "text": "<Region 1>: This region includes an individual who is caught in a moment that seems to involve some sort of task or activity. The person is engaged with a luggage cart, which suggests they might be arriving or departing from a location that offers such amenities, possibly a hotel. The cart holds luggage indicating travel or transit. The man's expression and attire provide clues to his role or state at the moment, such as potentially being a guest handling his luggage. The other individual seen partially in the background creates a sense of movement or interaction, but their relationship to the man or the context is unclear.\n"
}
predictions:
{
    "question_id": 1, 
    "image": "000000104486.jpg", 
    "category": "natural_box", 
    "text": "<Region 1>: The marked region does not appear to have any direct relationship with other marked regions, as there are no other marks to compare or contrast with.\n"
}
"""


def main(args):
    output_name = args.output_path.split("/")[-1]  # android_QA_box.json

    for phase in [
        "android_detailed_caption_box",
        "multipanel_detailed_caption_box",
        "natural_detailed_caption_box",
        "ocr_doc_detailed_caption_box",
        "ocr_spotting_detailed_caption_box",
        "web_detailed_caption_box",
    ]:
        vp = "bbox"
        domain = phase.split("_box")[0]  # android_QA

        if not os.path.exists(f"mdvp_for_gpt4v_eval/{phase}"):
            os.mkdir(f"mdvp_for_gpt4v_eval/{phase}")

        with open(args.output_path, "r") as f:
            data = json.load(f)

        with open("annotations/mdvp_caption_mask.json", "r") as f:
            mask_data = json.load(f)

        format_answer_list = []
        format_prediction_list = []
        for index, item in enumerate(data):
            meta = mask_data[index]
            assert meta["caption"] == item["gt"]

            try:
                image_path = item["image_path"]
            except:
                image_path = item["file_name"]

            format_answer = {
                "question_id": index + 1,
                "image": image_path,
                "category": meta["dataset_name"],
                "text": item["gt"],
            }
            format_answer_list.append(format_answer)

            format_prediction = {
                "question_id": index + 1,
                "image": image_path,
                "category": meta["dataset_name"],
                "text": item["caption"],
            }
            format_prediction_list.append(format_prediction)

        with open(f"mdvp_for_gpt4v_eval/{phase}/answer.json", "w") as f:
            json.dump(format_answer_list, f, indent=4, ensure_ascii=False)
        print(f"mdvp_for_gpt4v_eval/{phase}/answer.json saved successfully!")

        with open(f"mdvp_for_gpt4v_eval/{phase}/prediction.json", "w") as f:
            json.dump(format_prediction_list, f, indent=4, ensure_ascii=False)
        print(f"mdvp_for_gpt4v_eval/{phase}/prediction.json saved successfully!")

        with open(f"data/{domain}/{domain}_box.json", "r") as f:
            data = json.load(f)
        format_question_list = []
        for index, item in enumerate(data):
            format_question = {
                "question_id": index + 1,
                "image": item["image_name"],
                "category": phase,
                "text": item["question"],
                "annotation": {f"{vp}": item[f"{vp}"], "segmentation": []},
            }
            format_question_list.append(format_question)
        with open(f"mdvp_for_gpt4v_eval/{phase}/question.json", "w") as f:
            json.dump(format_question_list, f, indent=4, ensure_ascii=False)
        print(f"mdvp_for_gpt4v_eval/{phase}/question.json saved successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process args.")
    parser.add_argument(
        "--output_path", type=str, required=True, help="Path to output results"
    )
    args = parser.parse_args()

    main(args)
