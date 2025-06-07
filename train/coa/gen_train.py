import json
import os

RAW_DATASET_PATH = os.getenv("RAW_DATASET_PATH", "")

with open("cache/coa.json", "r") as f:
    data = json.load(f)

instruction = "你是一名安全审核员。经过分析后，将以下内容分类为[safe, unsafe]中的仅一类，你的回答应该以<ANALYSIS>...</ANALYSIS><ANALOGY>...</ANALOGY><REASONING>...</REASONING>的形式给出，其中ANALOGY部分专门负责类比参考案例，并在</REASONING>之后直接给出最终分类结果。"
outputs = []
for _data in data:
    output_text = _data["coa"].split("</think>")[-1]
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