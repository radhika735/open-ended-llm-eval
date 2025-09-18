import os 
import json
from pathlib import Path
from collections import defaultdict


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
    

# def get_num_duplicate_questions_in_qu_file(qu_filepath):
#     with open(qu_filepath, "r", encoding="utf-8") as f:
#         qu_data = json.load(f)
    
#     questions = [el["question"] for el in qu_data]
#     unique_questions = set(questions)
#     num_duplicates = len(questions) - len(unique_questions)
#     for qu in questions:
#         if questions.count(qu) > 1:
#             print(f"Duplicate question in {os.path.basename(qu_filepath)}: {qu}")
#     return num_duplicates



# def get_num_duplicate_questions_in_summary_file(summaries_filepath):
#     with open(summaries_filepath, "r", encoding="utf-8") as f:
#         summaries_data = json.load(f)
    
#     questions = [el["query"] for el in summaries_data]
#     unique_questions = set(questions)
#     num_duplicates = len(questions) - len(unique_questions)
#     for qu in questions:
#         if questions.count(qu) > 1:
#             print(f"Duplicate question in {os.path.basename(summaries_filepath)}: {qu}")
#     return num_duplicates



# def get_num_duplicate_questions(question_type, filter_stage, model, provider, usage_stage="usage_annotated", eval_stage="eval_annotated"):
#     cleaned_model_name = parse_model_name(model)
#     cleaned_provider_name = parse_provider_name(provider)
#     cleaned_name = f"{cleaned_provider_name}_{cleaned_model_name}"
#     summaries_dir = f"tool_unfailed_summaries/{question_type}_{filter_stage}_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}"
#     qus_dir = f"tool_unfailed_questions/bg_km_qus/{question_type}/{filter_stage}/{usage_stage}"

#     qu_dir_duplicates = 0
#     for qu_filename in sorted(os.listdir(qus_dir)):
#         qu_filepath = os.path.join(qus_dir, qu_filename)
#         qu_dir_duplicates += get_num_duplicate_questions_in_qu_file(qu_filepath)
#     print(f"Number of duplicate questions in {qus_dir}: {qu_dir_duplicates}")

#     summary_dir_duplicates = 0
#     for summaries_filename in sorted(os.listdir(summaries_dir)):
#         summaries_filepath = os.path.join(summaries_dir, summaries_filename)
#         summary_dir_duplicates += get_num_duplicate_questions_in_summary_file(summaries_filepath)
#     print(f"Number of duplicate questions in {summaries_dir}: {summary_dir_duplicates}")



# def get_num_duplicate_qus_across_files(path1, path2):
#     print("\n\n")
#     print(f"Path 1: {path1}")
#     print(f"Path 2: {path2}")
#     with open(path1, 'r', encoding="utf-8") as f:
#         data1 = json.load(f)
#     with open(path2, 'r', encoding="utf-8") as f:
#         data2 = json.load(f)
#     data2_queries = [el["query"] for el in data2]
#     data1_queries = [el["query"] for el in data1]
#     seen_duplicates = set()
#     for qu in data1_queries:
#         if qu in data2_queries:
#             seen_duplicates.add(qu)
#             print(f"Duplicate question found in path1 to path2: {qu}")
#     for qu in data2_queries:
#         if qu in data1_queries and qu not in seen_duplicates:
#             seen_duplicates.add(qu)
#             print(f"Duplicate question found in path2 to path1: {qu}")
#     print(f"Number of duplicate questions found: {len(seen_duplicates)}")
#     return len(seen_duplicates)



# def get_num_duplicate_qus_across_dirs(dir1, dir2):
#     total_duplicates = 0
#     done_files = set()
#     for filename in sorted(os.listdir(dir1)):
#         path1 = os.path.join(dir1, filename)
#         path2 = os.path.join(dir2, filename)
#         if os.path.exists(path2):
#             total_duplicates += get_num_duplicate_qus_across_files(path1, path2)
#             done_files.add(filename)
#         else:
#             print(f"File {filename} not found in {dir2}, skipping.")

#     for filename in sorted(os.listdir(dir2)):
#         if filename in done_files:
#             continue
#         else:
#             path2 = os.path.join(dir2, filename)
#             path1 = os.path.join(dir1, filename)
#             if os.path.exists(path1):
#                 total_duplicates += get_num_duplicate_qus_across_files(path1, path2)
#                 done_files.add(filename)
#             else:
#                 print(f"File {filename} not found in {dir1}, skipping.")

#     print(f"\n\n\nTotal number of duplicate questions across directories: {total_duplicates}")



def write_json_file(data_list, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            json.dump(data_list, file, indent=2)
    except TypeError as e:
        print(f"Error writing to JSON file {filepath}: {e}")



def split_summaries_file(all_summaries_filepath, failed_qus_filepath, unfailed_qus_filepath, failed_summaries_filepath, unfailed_summaries_filepath):
    with open(all_summaries_filepath, "r", encoding="utf-8") as f:
        all_summaries_details = json.load(f)

    if os.path.exists(failed_qus_filepath):
        with open(failed_qus_filepath, "r", encoding="utf-8") as f:
            failed_qus_details = json.load(f)
    else:
        failed_qus_details = []

    if os.path.exists(unfailed_qus_filepath):
        with open(unfailed_qus_filepath, "r", encoding="utf-8") as f:
            unfailed_qus_details = json.load(f)
    else:
        unfailed_qus_details = []

    tagged_queries = defaultdict(list)
    for qus_details in failed_qus_details:
        query = qus_details["question"]
        tagged_queries[query].append("failed")

    for qus_details in unfailed_qus_details:
        query = qus_details["question"]
        tagged_queries[query].append("unfailed")

    failed_summaries_details = []
    unfailed_summaries_details = []

    for summary_details in all_summaries_details:
        query = summary_details["query"]
        if "failed" in tagged_queries.get(query, []):
            # print("FAILED:", query)
            failed_summaries_details.append(summary_details)
        elif "unfailed" in tagged_queries.get(query, []):
            # print("UNFAILED:", query)
            unfailed_summaries_details.append(summary_details)
        else:
            print("NOT TAGGED:", query)
            
    write_json_file(failed_summaries_details, failed_summaries_filepath)
    write_json_file(unfailed_summaries_details, unfailed_summaries_filepath)



def split(qu_type, filter_stage, model, provider):
    cleaned_model_name = parse_model_name(model)
    cleaned_provider_name = parse_provider_name(provider)
    cleaned_name = f"{cleaned_provider_name}_{cleaned_model_name}"
    eval_stages = ["all_eval_stages","eval_annotated"]
    for eval_stage in eval_stages:
        all_summaries_dir = f"live_summaries/{qu_type}_{filter_stage}_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}"
        failed_summaries_dir = f"tool_failed_summaries/{qu_type}_{filter_stage}_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}"
        unfailed_summaries_dir = f"tool_unfailed_summaries/{qu_type}_{filter_stage}_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}"
        failed_qus_dir = f"tool_failed_questions/bg_km_qus/{qu_type}/{filter_stage}/usage_annotated"
        unfailed_qus_dir = f"tool_unfailed_questions/bg_km_qus/{qu_type}/{filter_stage}/usage_annotated"
        for summaries_filename in sorted(os.listdir(all_summaries_dir)):
            print("\n", summaries_filename)
            all_summaries_path = os.path.join(all_summaries_dir, summaries_filename)
            failed_summaries_path = os.path.join(failed_summaries_dir, summaries_filename)
            unfailed_summaries_path = os.path.join(unfailed_summaries_dir, summaries_filename)
            failed_qus_path = os.path.join(failed_qus_dir, summaries_filename.replace("summaries.json", "qus.json"))
            unfailed_qus_path = os.path.join(unfailed_qus_dir, summaries_filename.replace("summaries.json", "qus.json"))
            split_summaries_file(
                all_summaries_filepath=all_summaries_path,
                failed_qus_filepath=failed_qus_path,
                unfailed_qus_filepath=unfailed_qus_path,
                failed_summaries_filepath=failed_summaries_path,
                unfailed_summaries_filepath=unfailed_summaries_path
            )
        


def split_questions_file(all_qus_filepath, failed_qus_filepath, unfailed_qus_filepath):
    with open(all_qus_filepath, "r", encoding="utf-8") as f:
        all_qus_details = json.load(f)
    if os.path.exists(failed_qus_filepath):
        with open(failed_qus_filepath, "r", encoding="utf-8") as f:
            failed_qus_details = json.load(f)
    else:
        failed_qus_details = []
    failed_qus_queries = [el["question"] for el in failed_qus_details]
    unfailed_qus_details = []
    for qus_details in all_qus_details:
        query = qus_details["question"]
        if query in failed_qus_queries:
            print("FAILED:", query)
        else:
            print("UNFAILED:", query)
            unfailed_qus_details.append(qus_details)
    with open(unfailed_qus_filepath, "w", encoding="utf-8") as f:
        json.dump(unfailed_qus_details, f, indent=2, ensure_ascii=False)



def split_questions_dir(all_qus_dir, failed_qus_dir, unfailed_qus_dir):
    for all_filename in sorted(os.listdir(all_qus_dir)):
        print("\n", all_filename)
        all_qus_path = os.path.join(all_qus_dir, all_filename)
        failed_qus_path = os.path.join(failed_qus_dir, all_filename)
        unfailed_qus_path = os.path.join(unfailed_qus_dir, all_filename)
        split_questions_file(
            all_qus_filepath=all_qus_path,
            failed_qus_filepath=failed_qus_path,
            unfailed_qus_filepath=unfailed_qus_path
        )



def split_all_questions(qu_type, filter_stage):
    live_questions_dir = f"live_questions/bg_km_qus/{qu_type}/{filter_stage}/usage_annotated"
    failed_questions_dir = f"tool_failed_questions/bg_km_qus/{qu_type}/{filter_stage}/usage_annotated"
    unfailed_questions_dir = f"tool_unfailed_questions/bg_km_qus/{qu_type}/{filter_stage}/usage_annotated"
    os.makedirs(unfailed_questions_dir, exist_ok=True)
    split_questions_dir(
        all_qus_dir=live_questions_dir,
        failed_qus_dir=failed_questions_dir,
        unfailed_qus_dir=unfailed_questions_dir
    )



def get_number_items_in_file(filepath):
    try:
        with open(filepath, 'r', encoding="utf-8") as file:
            items = json.load(file) 
        return len(items)
    except FileNotFoundError:
        print(f"File {filepath} not found.")
        raise
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {filepath}.")
        raise



def get_number_items_in_dir_recursive(base_directory):
    total_items = 0
    # for root, dirs, files in os.walk(base_directory):
    #     for filename in sorted(files):
    #         if filename.endswith('.json'):
    #             filepath = os.path.join(root, filename)
    #             num_items = get_number_items_in_file(filepath)
    #             total_items += num_items
    # return total_items
    path = Path(base_directory)
    for p in path.rglob('*.json'):
        num_items = get_number_items_in_file(p)
        total_items += num_items
    return total_items


# def get_query_difference_between_files(qu_filepath, summary_filepath):
#     print("\n\n")
#     print(f"Qu path: {qu_filepath}")
#     print(f"Summary path: {summary_filepath}")
#     try:
#         with open(qu_filepath, 'r', encoding="utf-8") as f:
#             data1 = json.load(f)
#         with open(summary_filepath, 'r', encoding="utf-8") as f:
#             data2 = json.load(f)
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         raise
#     differences = []
#     data1_queries = [el["question"] for el in data1]
#     data2_queries = [el["query"] for el in data2]
#     for qu in data1_queries:
#         if qu not in data2_queries:
#             differences.append(qu)
#             print(f"Difference query found in qu path not in summary path: {qu}")
#     for qu in data2_queries:
#         if qu not in data1_queries:
#             differences.append(qu)
#             print(f"Difference query found in summary path not in qu path: {qu}")
#     print(f"Number of difference questions found: {len(differences)}")
#     return len(differences)


# def get_query_difference_between_dirs(qu_dir, summary_dir):
#     total_differences = 0
#     done_files = set()
#     for qu_filename in sorted(os.listdir(qu_dir)):
#         summary_filename = qu_filename.replace("qus.json", "summaries.json")
#         qu_path = os.path.join(qu_dir, qu_filename)
#         summary_path = os.path.join(summary_dir, summary_filename)
#         total_differences += get_query_difference_between_files(qu_filepath=qu_path, summary_filepath=summary_path)
#         done_files.add(qu_filename)
#         done_files.add(summary_filename)

#     for summary_filename in sorted(os.listdir(summary_dir)):
#         if summary_filename in done_files:
#             continue
#         else:
#             qu_filename = summary_filename.replace("summaries.json", "qus.json")
#             summary_path = os.path.join(summary_dir, summary_filename)
#             qu_path = os.path.join(qu_dir, qu_filename)
#             total_differences += get_query_difference_between_files(qu_filepath=qu_path, summary_filepath=summary_path)
#             done_files.add(qu_filename)
#             done_files.add(summary_filename)

#     print(f"\n\n\nTotal number of difference questions across directories: {total_differences}")



def main():
    MODEL_PROVIDER_LIST = [
        ("openai/gpt-5", None),
        ("anthropic/claude-sonnet-4", None),
        ("google/gemini-2.5-pro", None),
        ("moonshotai/kimi-k2-0905", "fireworks/fp8")
    ]

    # ## Splitting all summaries into tool failed and tool unfailed:
    # qu_types = ["answerable", "unanswerable"]
    # filter_stages = ["passed", "all"]
    # for qu_type in qu_types:
    #     for filter_stage in filter_stages:
    #         for model, provider in MODEL_PROVIDER_LIST:
    #             split(qu_type=qu_type, filter_stage=filter_stage, model=model, provider=provider)



    # ## Seeing if num unfailed questions and num failed questions add up to total questions:
    # num_tool_unfailed_questions = get_number_items_in_dir_recursive("tool_unfailed_questions")
    # num_tool_failed_questions = get_number_items_in_dir_recursive("tool_failed_questions")
    # live_qus_dirs = [
    #     f"live_questions/bg_km_qus/answerable/failed/usage_annotated", 
    #     f"live_questions/bg_km_qus/answerable/passed/usage_annotated",
    #     f"live_questions/bg_km_qus/unanswerable/failed/usage_annotated",
    #     f"live_questions/bg_km_qus/unanswerable/passed/usage_annotated"
    # ]
    # total_live_questions = 0
    # for dir in live_qus_dirs:
    #     total_live_questions += get_number_items_in_dir_recursive(dir)

    # print("Total number of tool unfailed questions:", num_tool_unfailed_questions)
    # print("Total number of tool failed questions:", num_tool_failed_questions)
    # print("Total number of live questions:", total_live_questions)



    # ## Seeing if num tool failed summaries and tool failed qus align:
    # num_tool_failed_questions = get_number_items_in_dir_recursive("tool_failed_questions")

    # eval_stage = "all_eval_stages"
    # m, p = MODEL_PROVIDER_LIST[3]
    # cleaned_m = parse_model_name(m)
    # cleaned_p = parse_provider_name(p)
    # cleaned_name = f"{cleaned_p}_{cleaned_m}"
    # live_summaries_dirs =[
    #     f"tool_failed_summaries/answerable_failed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}",
    #     f"tool_failed_summaries/answerable_passed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}",
    #     f"tool_failed_summaries/unanswerable_failed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}",
    #     f"tool_failed_summaries/unanswerable_passed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}"
    # ]
    # num_tool_failed_summaries = 0
    # for dir in live_summaries_dirs:
    #     num_tool_failed_summaries += get_number_items_in_dir_recursive(dir)

    # print("Total number of tool failed questions:", num_tool_failed_questions)
    # print(f"Total number of tool failed summaries ({cleaned_name}):", num_tool_failed_summaries)



    # ## Seeing if num tool unfailed summaries and tool unfailed qus align:
    # num_tool_unfailed_questions = get_number_items_in_dir_recursive("tool_unfailed_questions")

    # eval_stage = "all_eval_stages"
    # m, p = MODEL_PROVIDER_LIST[3]
    # cleaned_m = parse_model_name(m)
    # cleaned_p = parse_provider_name(p)
    # cleaned_name = f"{cleaned_p}_{cleaned_m}"
    # live_summaries_dirs =[
    #     f"tool_unfailed_summaries/answerable_failed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}",
    #     f"tool_unfailed_summaries/answerable_passed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}",
    #     f"tool_unfailed_summaries/unanswerable_failed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}",
    #     f"tool_unfailed_summaries/unanswerable_passed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}"
    # ]
    # num_tool_unfailed_summaries = 0
    # for dir in live_summaries_dirs:
    #     num_tool_unfailed_summaries += get_number_items_in_dir_recursive(dir)

    # print("Total number of tool unfailed questions:", num_tool_unfailed_questions)
    # print(f"Total number of tool unfailed summaries ({cleaned_name}):", num_tool_unfailed_summaries)



    ## Seeing if num live summaries and live qus align:
    live_qus_dirs = [
        f"live_questions/bg_km_qus/answerable/failed/usage_annotated",
        f"live_questions/bg_km_qus/answerable/passed/usage_annotated",
        f"live_questions/bg_km_qus/unanswerable/failed/usage_annotated",
        f"live_questions/bg_km_qus/unanswerable/passed/usage_annotated"
    ]
    num_live_questions = 0
    for dir in live_qus_dirs:
        num_live_questions += get_number_items_in_dir_recursive(dir)

    eval_stage = "all_eval_stages"
    m, p = MODEL_PROVIDER_LIST[3]
    cleaned_m = parse_model_name(m)
    cleaned_p = parse_provider_name(p)
    cleaned_name = f"{cleaned_p}_{cleaned_m}"
    live_summaries_dirs =[
        f"live_summaries/answerable_failed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}",
        f"live_summaries/answerable_passed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}",
        f"live_summaries/unanswerable_failed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}",
        f"live_summaries/unanswerable_passed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}"
    ]
    num_live_summaries = 0
    for dir in live_summaries_dirs:
        num_live_summaries += get_number_items_in_dir_recursive(dir)

    print("Total number of live questions:", num_live_questions)
    print(f"Total number of live summaries ({cleaned_name}):", num_live_summaries)







if __name__ == "__main__":
    main()