import json

with open(f"cache/desc_with_similarity.json", "r") as f:
    train = json.load(f)

path2train = {_train["image_path"]: _train for _train in train}

def build_refer(similarity_path):
    refer = "参考样本如下：\n"
    for _similarity_path in similarity_path:
        _sample = path2train[_similarity_path]
        refer += f"样本:{_sample['desc']};标注:{_sample['cot_with_label']}\n"
    return refer

data = []
for _path, _train in path2train.items():
    similarity_path = [_similarity[0] for _similarity in _train["similarity"]]
    refer = build_refer(similarity_path)
    _data = {"desc": _train["desc"], "refer": refer, "image_path": _path, "cot_without_label": _train["cot_without_label"], "cot_with_label": _train["cot_with_label"], "similarity_path": similarity_path}
    data.append(_data)

with open("cache/refer.json", "w") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)