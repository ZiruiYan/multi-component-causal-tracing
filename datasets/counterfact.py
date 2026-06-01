import json
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "counterfact" / "counterfact.json"


def get_factual(num=200, path=DATA_FILE):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_prompts = []
    target_new_list = []
    target_true_list = []

    for block in data:
        rewrite_info = block.get("requested_rewrite", {})
        prompt_template = rewrite_info.get("prompt", "")
        subject = rewrite_info.get("subject", "")

        formatted_prompts.append(prompt_template.format(subject))
        target_new_list.append(rewrite_info.get("target_new", {})["str"])
        target_true_list.append(rewrite_info.get("target_true", {})["str"])

    return formatted_prompts[:num], target_new_list[:num], target_true_list[:num]
