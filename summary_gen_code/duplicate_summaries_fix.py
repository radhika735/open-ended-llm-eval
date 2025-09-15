import os
import json
import logging
from collections import defaultdict
import copy

from utils.exceptions import RetrievalError



class Counter():
    def __init__(self):
        self.count = 0

    def increment(self, amount=1):
        self.count += amount

    def get_count(self):
        return self.count



def parse_model_name(model):
    model_split = model.split("/")
    model_name = model_split[-1]
    cleaned_name = ""
    for char in model_name:
        if char.isalnum():
            cleaned_name += char
        else:
            cleaned_name += "-"
    return cleaned_name



def parse_provider_name(provider):
    if provider is not None:
        provider_split = provider.split("/")
        provider_name = provider_split[0]
        return provider_name
    else:
        return ""
    


def read_json_file(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
            if not isinstance(data, list):
                raise RetrievalError(f"Expected JSON file {filepath} to contain a list, but contained {type(data)} instead.")
            else:
                return data 
    except json.JSONDecodeError as e:
        raise RetrievalError(f"Error reading json from file {filepath}: {str(e)}.")
    except FileNotFoundError:
        raise RetrievalError(f"File {filepath} not found.")
    


def write_json_file(data_list, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(data_list, file, indent=2)
    except TypeError as e:
        logging.error(f"Error writing to JSON file {filepath}: {e}")



def get_best_copy(summaries):
    reversed_summaries = summaries[::-1]

    best = None
    for summary in reversed_summaries:
        if summary["relevant_summary"] is not None:
            best = summary
            break
    if best is None:
        best = reversed_summaries[0]

    return best


def get_deduplicated_summaries_for_file(summaries_filepath):
    deduplicated_dicts = defaultdict(list)

    summaries_data = read_json_file(summaries_filepath)
    summaries_queries = [su["query"] for su in summaries_data]

    for query, full_dict in zip(summaries_queries, summaries_data):
        deduplicated_dicts[query].append(full_dict)

    deduplicated_ordered = []
    seen_duplicates = set()
    for query in summaries_queries:
        if len(deduplicated_dicts[query]) == 0:
            logging.error(f"Query '{query}' not found in deduplicated dicts.")
        elif len(deduplicated_dicts[query]) > 1:
            if query not in seen_duplicates:
                logging.info(f"{len(deduplicated_dicts[query])} summaries found for query: {query}")
                seen_duplicates.add(query)
                best_summary = get_best_copy(deduplicated_dicts[query])
                deduplicated_ordered.append(best_summary)
            else:
                pass
        else:
            deduplicated_ordered.append(deduplicated_dicts[query][0])

    return deduplicated_ordered



def handle_overlapping_summaries_for_file(summaries_filepath):
    deduplicated = get_deduplicated_summaries_for_file(summaries_filepath)
    # copy_filepath = os.path.splitext(summaries_filepath)[0] + "_dedupcopy.json"
    copy_filepath = summaries_filepath
    logging.warning(f"Writing deduplicated summaries to file {copy_filepath}...")
    write_json_file(data_list=deduplicated, filepath=copy_filepath)



def handle_overlapping_summaries_for_dir(model, provider, qu_type, filter_stage):
    cleaned_model_name = parse_model_name(model)
    cleaned_provider_name = parse_provider_name(provider)
    cleaned_name = f"{cleaned_provider_name}_{cleaned_model_name}"
    summaries_dir = f"live_summaries/{qu_type}_{filter_stage}_qus_summaries/hybrid_cross-encoder/all_eval_stages/{cleaned_name}"

    for filename in sorted(os.listdir(summaries_dir)):
        if filename.endswith(".json"):
            summaries_filepath = os.path.join(summaries_dir, filename)
            handle_overlapping_summaries_for_file(summaries_filepath=summaries_filepath)



def get_num_overlapping_summaries_for_file(summaries_filepath):
    deduplicated = get_deduplicated_summaries_for_file(summaries_filepath)
    original = read_json_file(summaries_filepath)
    num_overlapping = len(original) - len(deduplicated)
    return num_overlapping



def get_num_overlapping_summaries_for_dir(model, provider, qu_type, filter_stage):
    cleaned_model_name = parse_model_name(model)
    cleaned_provider_name = parse_provider_name(provider)
    cleaned_name = f"{cleaned_provider_name}_{cleaned_model_name}"
    summaries_dir = f"live_summaries/{qu_type}_{filter_stage}_qus_summaries/hybrid_cross-encoder/all_eval_stages/{cleaned_name}"

    total_overlapping = 0
    for filename in sorted(os.listdir(summaries_dir)):
        if filename.endswith(".json"):
            summaries_filepath = os.path.join(summaries_dir, filename)
            num_overlapping = get_num_overlapping_summaries_for_file(summaries_filepath=summaries_filepath)
            if num_overlapping > 0:
                logging.info(f"{num_overlapping} overlapping summaries found in file {summaries_filepath}.")
            total_overlapping += num_overlapping

    if total_overlapping == 0:
        logging.info(f"No overlapping summaries found in directory {summaries_dir}.")
    else:
        logging.info(f"Total of {total_overlapping} overlapping summaries found in directory {summaries_dir}.")

    return total_overlapping


# def del_old_files(model, provider, qu_type, filter_stage):
    # cleaned_model_name = parse_model_name(model)
    # cleaned_provider_name = parse_provider_name(provider)
    # cleaned_name = f"{cleaned_provider_name}_{cleaned_model_name}"
    # summaries_dir = f"live_summaries/{qu_type}_{filter_stage}_qus_summaries/hybrid_cross-encoder/all_eval_stages/{cleaned_name}"

    # for filename in sorted(os.listdir(summaries_dir)):
    #     if filename.endswith(".json") and not filename.endswith("_dedupcopy.json"):
    #         summaries_filepath = os.path.join(summaries_dir, filename)
    #         logging.warning(f"Deleting old file {summaries_filepath}...")
    #         os.remove(summaries_filepath)



# def rename_dedup_files(model, provider, qu_type, filter_stage):
#     cleaned_model_name = parse_model_name(model)
#     cleaned_provider_name = parse_provider_name(provider)
#     cleaned_name = f"{cleaned_provider_name}_{cleaned_model_name}"
#     summaries_dir = f"live_summaries/{qu_type}_{filter_stage}_qus_summaries/hybrid_cross-encoder/all_eval_stages/{cleaned_name}"

#     for filename in sorted(os.listdir(summaries_dir)):
#         if filename.endswith("_dedupcopy.json"):
#             new_filename = filename.replace("_dedupcopy", "")
#             new_filepath = os.path.join(summaries_dir, new_filename)
#             old_filepath = os.path.join(summaries_dir, filename)
#             logging.warning(f"Renaming file {filename} to {new_filename}...")
#             os.rename(old_filepath, new_filepath)
            




def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    MODEL_PROVIDER_LIST = [
        ("openai/gpt-5", None),
        ("anthropic/claude-sonnet-4", None),
        ("google/gemini-2.5-pro", None),
        ("moonshotai/kimi-k2-0905", "fireworks/fp8")
    ]

    qu_types = ["answerable", "unanswerable"]
    filter_stages = ["passed", "failed"]
    # unanswerable passed and answerable passed both have duplicates.

    for m, p in MODEL_PROVIDER_LIST:
        for qu_type in qu_types:
            for filter_stage in filter_stages:
                get_num_overlapping_summaries_for_dir(model=m, provider=p, qu_type=qu_type, filter_stage=filter_stage)


    # model, provider = MODEL_PROVIDER_LIST[0]

    # cleaned_model_name = parse_model_name(model)
    # cleaned_provider_name = parse_provider_name(provider)
    # cleaned_name = f"{cleaned_provider_name}_{cleaned_model_name}"
    # summaries_filepath=f"live_summaries/{qu_type}_{filter_stage}_qus_summaries/hybrid_cross-encoder/all_eval_stages/{cleaned_name}/bg_km_PrimateConservation_summaries.json"

    # get_overlapping_summaries_for_file(summaries_filepath=summaries_filepath)
    # handle_overlapping_summaries_for_dir(model=model, provider=provider, qu_type=qu_type, filter_stage=filter_stage)


if __name__ == "__main__":
    main()