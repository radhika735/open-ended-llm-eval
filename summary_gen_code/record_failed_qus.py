import os
import json

def store_failed_qus(qu_type, filter_stage):
    orig_base_dir = "live_questions"
    failed_base_dir = "failed_questions"
    remaining_dir = f"bg_km_qus/{qu_type}/{filter_stage}/usage_annotated"

    orig_dir = os.path.join(orig_base_dir, remaining_dir)
    for filename in sorted(os.listdir(orig_dir)):
        filepath = os.path.join(orig_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        failed_qus = []
        for qu in questions:
            if qu.get("used_by_models", []) != []:
                failed_qus.append(qu)

        failed_dir = os.path.join(failed_base_dir, remaining_dir)
        os.makedirs(failed_dir, exist_ok=True)
        with open(os.path.join(failed_dir, filename), 'w', encoding='utf-8') as f:
            json.dump(failed_qus, f, indent=2)

def main():
    qu_types = ["answerable", "unanswerable"]
    filter_stages = ["passed", "failed"]
    for type in qu_types:
        for stage in filter_stages:
            store_failed_qus(qu_type=type, filter_stage=stage)

if __name__ == "__main__":
    pass