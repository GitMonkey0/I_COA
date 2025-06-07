import json
from utils import VLLM
import os

MODEL_PATH = os.getenv("MODEL_PATH", "")

def main():
    with open("cache/refer.json", "r") as f:
        data = json.load(f)
    llm = VLLM(model=f"{MODEL_PATH}/QwQ-32B")
    system_prompt = "你是一名安全审核员。经过分析后，将以下内容分类为[safe, unsafe]中的仅一类，你的回答应该以<ANALYSIS>...</ANALYSIS><ANALOGY>...</ANALOGY><REASONING>...</REASONING>的形式给出，其中ANALOGY部分专门负责类比参考案例，并在</REASONING>之后直接给出最终分类结果。"
    prompts = [f'''案例:{_data["refer"]}\n\n要分析的句子:{_data["desc"]};人类对他的标注是:{_data["cot_with_label"]}''' for _data in data]
    outputs = llm.generate(prompts, system_prompt=system_prompt, enable_thinking=True)
    for _data, output in zip(data, outputs):
        _data.update({"coa": output})

    with open("cache/coa.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()