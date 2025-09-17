import os
import json

def reset_usage_in_failed_summaries(qus_dir):
    for root, _, files in os.walk(qus_dir):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                print(file_path)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Reset usage annotations
                for qu_dict in data:
                    qu_dict["used_by_models"] = []
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)


def create_all_usage_levels_dirs(base_dir="tool_failed_questions"):
    for root, dirs, _ in os.walk(base_dir):
        for dir_name in dirs:
            if dir_name == "usage_annotated":
                annotated_usage_dir_path = os.path.join(root, dir_name)
                all_usage_dir_path = os.path.join(root, "all_usage_stages_for_summary_gen")
                os.makedirs(all_usage_dir_path, exist_ok=True)
                print(f"Created directory: {all_usage_dir_path}")
                for qu_file in sorted(os.listdir(annotated_usage_dir_path)):
                    with open(os.path.join(annotated_usage_dir_path, qu_file), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    for qu_dict in data:
                        qu_dict.pop("used_by_models", None)
                    with open(os.path.join(all_usage_dir_path, qu_file), 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)




def main():
    create_all_usage_levels_dirs("tool_unfailed_questions")


if __name__ == "__main__":
    main()