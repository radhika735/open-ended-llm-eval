import os
import logging
import json

def read_file(filepath):
    if os.path.exists(filepath):
        with open(filepath, 'w', encoding="utf-8") as f:
            return f.read()
    else:
        logging.warning(f"File {filepath} not found.")


def read_json_file(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, 'w', encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Could not load json from file: {filepath} .")
            return ""
    else:
        logging.warning(f"File {filepath} not found.")


def write_json_file(data, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def update_json_file(filepath, new_obj):
    current_obj = read_json_file(filepath)
    current_obj.update(new_obj)
    write_json_file(current_obj, filepath)


def get_token_count_for_action(action_base_dir, action_num, read_cache=False, write_cache=True, cache_filepath=""):
    filename = f"action_{action_num}_clean.txt"
    filepath = os.path.join(action_base_dir, filename)

    if read_cache:
        cache_data = read_json_file(cache_filepath)
        if cache_data != "" and (filename in cache_data):
            num_tokens = cache_data[filename]
            return num_tokens
        
    content = read_file(filepath)
    tokens = content.split()
    num_tokens = len(tokens)

    if write_cache:
        obj = {action_num: num_tokens}
        update_json_file(filepath=cache_filepath, new_obj=obj)

    return num_tokens



def get_all_token_counts(action_base_dir, read_cache=False, write_cache=False, cache_filepath=""):
    token_counts = {}

    if read_cache():
        cache_data = read_json_file(filepath=cache_filepath)
        if cache_data != "":
            token_counts = cache_data

    for entry in os.scandir(action_base_dir):
        action_filename = entry.name

        # Is a string:
        action_num_as_str = action_filename.split("_")[1]

        # Is an integer:
        num_tokens = get_token_count_for_action(action_base_dir=action_base_dir, action_num=action_num_as_str, read_cache=read_cache, write_cache=write_cache, cache_filepath=cache_filepath)
        token_counts[action_num_as_str] = num_tokens
    return token_counts