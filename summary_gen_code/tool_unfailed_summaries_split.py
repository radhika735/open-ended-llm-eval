import os 
import json
from pathlib import Path




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
    

def get_num_duplicate_questions_in_qu_file(qu_filepath):
    with open(qu_filepath, "r", encoding="utf-8") as f:
        qu_data = json.load(f)
    
    questions = [el["question"] for el in qu_data]
    unique_questions = set(questions)
    num_duplicates = len(questions) - len(unique_questions)
    for qu in questions:
        if questions.count(qu) > 1:
            print(f"Duplicate question in {os.path.basename(qu_filepath)}: {qu}")
    return num_duplicates



def get_num_duplicate_questions_in_summary_file(summaries_filepath):
    with open(summaries_filepath, "r", encoding="utf-8") as f:
        summaries_data = json.load(f)
    
    questions = [el["query"] for el in summaries_data]
    unique_questions = set(questions)
    num_duplicates = len(questions) - len(unique_questions)
    for qu in questions:
        if questions.count(qu) > 1:
            print(f"Duplicate question in {os.path.basename(summaries_filepath)}: {qu}")
    return num_duplicates



def get_num_duplicate_questions(question_type, filter_stage, model, provider, usage_stage="usage_annotated", eval_stage="eval_annotated"):
    cleaned_model_name = parse_model_name(model)
    cleaned_provider_name = parse_provider_name(provider)
    cleaned_name = f"{cleaned_provider_name}_{cleaned_model_name}"
    summaries_dir = f"tool_unfailed_summaries/{question_type}_{filter_stage}_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}"
    qus_dir = f"tool_unfailed_questions/bg_km_qus/{question_type}/{filter_stage}/{usage_stage}"

    qu_dir_duplicates = 0
    for qu_filename in sorted(os.listdir(qus_dir)):
        qu_filepath = os.path.join(qus_dir, qu_filename)
        qu_dir_duplicates += get_num_duplicate_questions_in_qu_file(qu_filepath)
    print(f"Number of duplicate questions in {qus_dir}: {qu_dir_duplicates}")

    summary_dir_duplicates = 0
    for summaries_filename in sorted(os.listdir(summaries_dir)):
        summaries_filepath = os.path.join(summaries_dir, summaries_filename)
        summary_dir_duplicates += get_num_duplicate_questions_in_summary_file(summaries_filepath)
    print(f"Number of duplicate questions in {summaries_dir}: {summary_dir_duplicates}")



def get_num_duplicate_qus_across_files(path1, path2):
    print("\n\n")
    print(f"Path 1: {path1}")
    print(f"Path 2: {path2}")
    with open(path1, 'r', encoding="utf-8") as f:
        data1 = json.load(f)
    with open(path2, 'r', encoding="utf-8") as f:
        data2 = json.load(f)
    data2_queries = [el["query"] for el in data2]
    data1_queries = [el["query"] for el in data1]
    seen_duplicates = set()
    for qu in data1_queries:
        if qu in data2_queries:
            seen_duplicates.add(qu)
            print(f"Duplicate question found in path1 to path2: {qu}")
    for qu in data2_queries:
        if qu in data1_queries and qu not in seen_duplicates:
            seen_duplicates.add(qu)
            print(f"Duplicate question found in path2 to path1: {qu}")
    print(f"Number of duplicate questions found: {len(seen_duplicates)}")
    return len(seen_duplicates)



def get_num_duplicate_qus_across_dirs(dir1, dir2):
    total_duplicates = 0
    done_files = set()
    for filename in sorted(os.listdir(dir1)):
        path1 = os.path.join(dir1, filename)
        path2 = os.path.join(dir2, filename)
        if os.path.exists(path2):
            total_duplicates += get_num_duplicate_qus_across_files(path1, path2)
            done_files.add(filename)
        else:
            print(f"File {filename} not found in {dir2}, skipping.")

    for filename in sorted(os.listdir(dir2)):
        if filename in done_files:
            continue
        else:
            path2 = os.path.join(dir2, filename)
            path1 = os.path.join(dir1, filename)
            if os.path.exists(path1):
                total_duplicates += get_num_duplicate_qus_across_files(path1, path2)
                done_files.add(filename)
            else:
                print(f"File {filename} not found in {dir1}, skipping.")

    print(f"\n\n\nTotal number of duplicate questions across directories: {total_duplicates}")




def split_summaries_file(all_filepath, failed_filepath, unfailed_filepath):
    with open(all_filepath, "r", encoding="utf-8") as f:
        all_summary_details = json.load(f)

    if os.path.exists(failed_filepath):
        with open(failed_filepath, "r", encoding="utf-8") as f:
            failed_summaries = json.load(f)
    else:
        failed_summaries = []

    failed_queries = [s["query"] for s in failed_summaries]

    unfailed_summaries = []

    for summary_details in all_summary_details:
        query = summary_details["query"]
        if query in failed_queries:
            print("FAILED:",query)
        else:
            print("UNFAILED:",query)
            unfailed_summaries.append(summary_details)

    with open(unfailed_filepath, "w", encoding="utf-8") as f:
        json.dump(unfailed_summaries, f, indent=2, ensure_ascii=False)



def split_summaries_dir(question_type, filter_stage, retrieval_type, eval_stage, model, provider):
    cleaned_model_name = parse_model_name(model)
    cleaned_provider_name = parse_provider_name(provider)
    cleaned_name = f"{cleaned_provider_name}_{cleaned_model_name}"
    common_dir = f"{question_type}_{filter_stage}_qus_summaries/{retrieval_type}/{eval_stage}/{cleaned_name}"
    all_dir = os.path.join("live_summaries", common_dir)
    if not os.path.exists(all_dir):
        print(f"Directory {all_dir} does not exist.")
        return
    failed_dir = os.path.join("tool_failed_summaries", common_dir)
    unfailed_dir = os.path.join("tool_unfailed_summaries", common_dir)
    os.makedirs(unfailed_dir, exist_ok=True)

    for summaries_filename in sorted(os.listdir(all_dir)):
        print("\n",common_dir, summaries_filename)
        all_summaries_path = os.path.join(all_dir, summaries_filename)
        failed_summaries_path = os.path.join(failed_dir, summaries_filename)
        unfailed_summaries_path = os.path.join(unfailed_dir, summaries_filename)
        split_summaries_file(
            all_filepath=all_summaries_path,
            failed_filepath=failed_summaries_path,
            unfailed_filepath=unfailed_summaries_path
        )
        


def split_all_summaries(model_provider_list):
    question_type_list = ["answerable", "unanswerable"]
    filter_stages = ["failed", "passed"]
    retrieval_types = ["hybrid_cross-encoder"]
    eval_stages = ["all_eval_stages", "eval_annotated"]
    for qu_type in question_type_list:
        for filter_stage in filter_stages:
            for retrieval_type in retrieval_types:
                for eval_stage in eval_stages:
                    for model, provider in model_provider_list:
                        print(f"\nProcessing: {qu_type}, {filter_stage}, {retrieval_type}, {eval_stage}, {model}, {provider}")
                        split_summaries_dir(
                            question_type=qu_type,
                            filter_stage=filter_stage,
                            retrieval_type=retrieval_type,
                            eval_stage=eval_stage,
                            model=model,
                            provider=provider
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


def get_query_difference_between_files(qu_filepath, summary_filepath):
    print("\n\n")
    print(f"Qu path: {qu_filepath}")
    print(f"Summary path: {summary_filepath}")
    try:
        with open(qu_filepath, 'r', encoding="utf-8") as f:
            data1 = json.load(f)
        with open(summary_filepath, 'r', encoding="utf-8") as f:
            data2 = json.load(f)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        raise
    differences = []
    data1_queries = [el["question"] for el in data1]
    data2_queries = [el["query"] for el in data2]
    for qu in data1_queries:
        if qu not in data2_queries:
            differences.append(qu)
            print(f"Difference query found in qu path not in summary path: {qu}")
    for qu in data2_queries:
        if qu not in data1_queries:
            differences.append(qu)
            print(f"Difference query found in summary path not in qu path: {qu}")
    print(f"Number of difference questions found: {len(differences)}")
    return len(differences)


def get_query_difference_between_dirs(qu_dir, summary_dir):
    total_differences = 0
    done_files = set()
    for qu_filename in sorted(os.listdir(qu_dir)):
        summary_filename = qu_filename.replace("qus.json", "summaries.json")
        qu_path = os.path.join(qu_dir, qu_filename)
        summary_path = os.path.join(summary_dir, summary_filename)
        total_differences += get_query_difference_between_files(qu_filepath=qu_path, summary_filepath=summary_path)
        done_files.add(qu_filename)
        done_files.add(summary_filename)

    for summary_filename in sorted(os.listdir(summary_dir)):
        if summary_filename in done_files:
            continue
        else:
            qu_filename = summary_filename.replace("summaries.json", "qus.json")
            summary_path = os.path.join(summary_dir, summary_filename)
            qu_path = os.path.join(qu_dir, qu_filename)
            total_differences += get_query_difference_between_files(qu_filepath=qu_path, summary_filepath=summary_path)
            done_files.add(qu_filename)
            done_files.add(summary_filename)

    print(f"\n\n\nTotal number of difference questions across directories: {total_differences}")






def main():
    MODEL_PROVIDER_LIST = [
        ("openai/gpt-5", None),
        ("anthropic/claude-sonnet-4", None),
        ("google/gemini-2.5-pro", None),
        ("moonshotai/kimi-k2-0905", "fireworks/fp8")
    ]

    # model, provider = MODEL_PROVIDER_LIST[0]

    # question_type = "answerable"
    # filter_stage = "failed"
    # retrieval_type = "hybrid_cross-encoder"
    # usage_stage = "all_usage_stages_for_summary_gen"
    # eval_stage = "all_eval_stages"
    # get_num_duplicate_questions(
    #     question_type=question_type,
    #     filter_stage=filter_stage,
    #     retrieval_type=retrieval_type,
    #     usage_stage=usage_stage,
    #     eval_stage=eval_stage,
    #     model=model,
    #     provider=provider
    # )


    # split(
    #     question_type=question_type,
    #     filter_stage=filter_stage,
    #     retrieval_type=retrieval_type,
    #     eval_stage=eval_stage,
    #     model=model,
    #     provider=provider
    # )


    # split_all(MODEL_PROVIDER_LIST)


    ## Checking number of questions and summaries align:
    # usage_stage = "usage_annotated"
    # eval_stage = "eval_annotated"

    # live_qus_dirs = [
    #     f"live_questions/bg_km_qus/answerable/failed/{usage_stage}", 
    #     f"live_questions/bg_km_qus/answerable/passed/{usage_stage}",
    #     f"live_questions/bg_km_qus/unanswerable/failed/{usage_stage}",
    #     f"live_questions/bg_km_qus/unanswerable/passed/{usage_stage}"
    # ]

    # live_summaries_dirs = []
    # tool_failed_summaries_dirs = []
    # tool_unfailed_summaries_dirs = []
    # for m, p in MODEL_PROVIDER_LIST:
    #     cleaned_m = parse_model_name(m)
    #     cleaned_p = parse_provider_name(p)
    #     cleaned_name = f"{cleaned_p}_{cleaned_m}"

    #     live_summaries_dirs.append([
    #         f"live_summaries/answerable_failed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}",
    #         f"live_summaries/answerable_passed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}",
    #         f"live_summaries/unanswerable_failed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}",
    #         f"live_summaries/unanswerable_passed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}"
    #     ])
    #     tool_failed_summaries_dirs.append([
    #         f"tool_failed_questions/bg_km_qus/answerable/failed/usage_annotated",
    #         f"tool_failed_questions/bg_km_qus/answerable/passed/usage_annotated",
    #         f"tool_failed_questions/bg_km_qus/unanswerable/failed/usage_annotated",
    #         f"tool_failed_questions/bg_km_qus/unanswerable/passed/usage_annotated"
    #     ])
    #     tool_unfailed_summaries_dirs.append([
    #         f"tool_unfailed_summaries/answerable_failed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}",
    #         f"tool_unfailed_summaries/answerable_passed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}",
    #         f"tool_unfailed_summaries/unanswerable_failed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}",
    #         f"tool_unfailed_summaries/unanswerable_passed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}"
    #     ])

    # base_dirs = [live_summaries_dirs[0], tool_failed_summaries_dirs[0], tool_unfailed_summaries_dirs[0]]

    # for dir_set in base_dirs:
    #     all_dirs_total = 0
    #     for dir in dir_set:
    #         total_items = get_number_items_in_dir_recursive(dir)
    #         all_dirs_total += total_items
    #         print(f"Total number of items in {dir}: {total_items}")
    #     print("ALL DIRS TOTAL:", all_dirs_total)


    # ## Debugging problematic failed and unfailed num summaries not adding up:
    # dir1 = f"tool_failed_summaries/answerable_failed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}"
    # dir2 = f"tool_unfailed_summaries/answerable_failed_qus_summaries/hybrid_cross-encoder/{eval_stage}/{cleaned_name}"
    # get_num_duplicate_qus_across_dirs(dir1, dir2)


    # Splitting questions into failed and unfailed:
    # split_all_questions(qu_type="answerable", filter_stage="failed")
    # split_all_questions(qu_type="answerable", filter_stage="passed")
    # split_all_questions(qu_type="unanswerable", filter_stage="failed")
    # split_all_questions(qu_type="unanswerable", filter_stage="passed")


    # # Seeing if num unfailed questions and num failed questions add up to total questions:
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


    # Seeing if num live summaries and num live qus align:
    # num_live_questions = get_number_items_in_dir_recursive("tool_failed_questions")
    live_qus_dirs = [
        f"live_questions/bg_km_qus/answerable/failed/usage_annotated",
        f"live_questions/bg_km_qus/answerable/passed/usage_annotated",
        f"live_questions/bg_km_qus/unanswerable/failed/usage_annotated",
        f"live_questions/bg_km_qus/unanswerable/passed/usage_annotated"
    ]
    num_live_questions = 0
    for dir in live_qus_dirs:
        num_live_questions += get_number_items_in_dir_recursive(dir)

    eval_stage = "eval_annotated"
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


    # ## Seeing if there are duplicates in the tool unfailed questions:
    # for qu_type in ["answerable", "unanswerable"]:
    #     for filter_stage in ["failed", "passed"]:
    #         for model, provider in MODEL_PROVIDER_LIST:
    #             print(f"\nChecking duplicates for: {qu_type}, {filter_stage}, {model}, {provider}")
    #             get_num_duplicate_questions(qu_type, filter_stage, model, provider)


    ## Finding difference between unfailed questions and unfailed summaries:
    # get_query_difference_between_dirs(
    #     qu_dir="tool_unfailed_questions/bg_km_qus/answerable/failed/usage_annotated",
    #     summary_dir="tool_unfailed_summaries/answerable_failed_qus_summaries/hybrid_cross-encoder/eval_annotated/_claude-sonnet-4"
    # )


    ## Comparing each file at a time to get the differences
    # qus_file = "tool_"




if __name__ == "__main__":
    main()