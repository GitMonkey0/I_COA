import json
import os

RAW_DATASET_PATH = os.getenv("RAW_DATASET_PATH", "")

with open("cache/desc_with_similarity.json", "r") as f:
    data = json.load(f)

instruction = "你是一名安全审核员。经过分析后，将以下内容分类为[safe, unsafe]中的仅一类，你的回答应该以<REASONING>...</REASONING>的形式给出，并在</REASONING>之后直接给出最终分类结果。"
outputs = []
for _data in data:
    output_text = f'''<REASONING>{_data["cot_with_label"]}</REASONING>'''
    messages = [
        {
            "role": "user",
            "content": f"<image>{instruction}"
        },
        {
            "role": "assistant",
            "content": output_text
        }
    ]
    images = [f"{RAW_DATASET_PATH}/train_images/{_data['image_path']}"]
    output = {"messages": messages, "images": images}
    outputs.append(output)

with open("cache/train.json", "w") as f:
    json.dump(outputs, f, ensure_ascii=False, indent=2)