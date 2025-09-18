import os 
import json
from pathlib import Path


def rejoin_file(tool_failed_regen_summaries_filepath, tool_unfailed_summaries_filepath, output_filepath):
    try:
        with open(tool_failed_regen_summaries_filepath, 'r', encoding='utf-8') as f:
            tool_failed_summaries = json.load(f)
    except FileNotFoundError:
        tool_failed_summaries = []

    try:
        with open(tool_unfailed_summaries_filepath, 'r', encoding='utf-8') as f:
            tool_unfailed_summaries = json.load(f)
    except FileNotFoundError:
        tool_unfailed_summaries = []
    
    all_summaries = tool_failed_summaries + tool_unfailed_summaries

    output_dir = os.path.dirname(output_filepath)
    os.makedirs(output_dir, exist_ok=True)
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, indent=2)


def rejoin_all_files(current_all_dir, tool_failed_regen_dir, tool_unfailed_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(current_all_dir):
        for i, filename in enumerate(files):
            if filename.endswith('.json'):
                relative_path = os.path.relpath(root, current_all_dir)
                tool_failed_regen_filepath = os.path.join(tool_failed_regen_dir, relative_path, filename)
                tool_unfailed_filepath = os.path.join(tool_unfailed_dir, relative_path, filename)
                output_filepath = os.path.join(output_dir, relative_path, filename)
                try:
                    num_tool_failed_regen = get_number_items_in_file(tool_failed_regen_filepath)
                except FileNotFoundError:
                    num_tool_failed_regen = 0
                try:
                    num_tool_unfailed = get_number_items_in_file(tool_unfailed_filepath)
                except FileNotFoundError:
                    num_tool_unfailed = 0
                
                print(f"\n\n\n{tool_failed_regen_filepath} size: {num_tool_failed_regen}")
                print(f"{tool_unfailed_filepath} size: {num_tool_unfailed}")
                rejoin_file(
                    tool_failed_regen_summaries_filepath=tool_failed_regen_filepath,
                    tool_unfailed_summaries_filepath=tool_unfailed_filepath,
                    output_filepath=output_filepath
                )
                print(f"{output_filepath} size: {get_number_items_in_file(output_filepath)}")
                if get_number_items_in_file(output_filepath) != (num_tool_failed_regen + num_tool_unfailed):
                    raise ValueError(f"Mismatch in counts for {output_filepath}")



def get_number_items_in_file(filepath):
    try:
        with open(filepath, 'r', encoding="utf-8") as file:
            items = json.load(file) 
        return len(items)
    except FileNotFoundError:
        #print(f"File {filepath} not found.")
        raise
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {filepath}.")
        raise


def get_number_items_in_dir_recursive(base_directory):
    total_items = 0
    path = Path(base_directory)
    for p in path.rglob('*.json'):
        num_items = get_number_items_in_file(p)
        total_items += num_items
    return total_items


def main():
    # rejoin_all_files(
    #     current_all_dir="live_summaries",
    #     tool_failed_regen_dir="tool_failed_summaries_regenerated", 
    #     tool_unfailed_dir="tool_unfailed_summaries",
    #     output_dir="all_summaries_joined"
    # )
    print(f"all summaries joined size: {get_number_items_in_dir_recursive("all_summaries_joined")}")
    print(f"live summaries size: {get_number_items_in_dir_recursive("live_summaries")}")


if __name__ == "__main__":
    main()