import os 
import json




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
    return num_duplicates



def get_num_duplicate_questions_in_summary_file(summaries_filepath):
    with open(summaries_filepath, "r", encoding="utf-8") as f:
        summaries_data = json.load(f)
    
    questions = [el["query"] for el in summaries_data]
    unique_questions = set(questions)
    num_duplicates = len(questions) - len(unique_questions)
    return num_duplicates


def get_num_duplicate_questions(question_type, filter_stage, retrieval_type, eval_stage, model, provider):
    cleaned_model_name = parse_model_name(model)
    cleaned_provider_name = parse_provider_name(provider)
    cleaned_name = f"{cleaned_provider_name}_{cleaned_model_name}"
    summaries_dir = f"live_summaries/{question_type}_{filter_stage}_qus_summaries/{retrieval_type}/{eval_stage}/{cleaned_name}"
    qu_dir = os.path.join("live_questions", common_dir)

    for qu_filename in sorted(os.listdir(qu_dir)):
        print(qu_filename)
        qu_filepath = os.path.join(qu_dir, qu_filename)
        num_qu_duplicates = get_num_duplicate_questions_in_qu_file(qu_filepath)
        print(f"Number of duplicate questions in {qu_filename}: {num_qu_duplicates}")

    for summaries_filename in sorted(os.listdir(summaries_dir)):
        print(summaries_filename)
        summaries_filepath = os.path.join(summaries_dir, summaries_filename)
        num_summary_duplicates = get_num_duplicate_questions_in_summary_file(summaries_filepath)
        print(f"Number of duplicate questions in {summaries_filename}: {num_summary_duplicates}")



def split(question_type, filter_stage, retrieval_type, eval_stage, model, provider):
    cleaned_model_name = parse_model_name(model)
    cleaned_provider_name = parse_provider_name(provider)
    cleaned_name = f"{cleaned_provider_name}_{cleaned_model_name}"
    common_dir = f"{question_type}_{filter_stage}_qus_summaries/{retrieval_type}/{eval_stage}/{cleaned_name}"
    all_dir = os.path.join("live_summaries", common_dir)
    failed_dir = os.path.join("tool_failed_summaries", common_dir)
    unfailed_dir = os.path.join("tool_unfailed_summaries", common_dir)
    os.makedirs(unfailed_dir, exist_ok=True)

    for summaries_filename in sorted(os.listdir(all_dir)):
        print(summaries_filename)
        all_summaries_path = os.path.join(all_dir, summaries_filename)
        failed_summaries_path = os.path.join(failed_dir, summaries_filename)
        unfailed_summaries_path = os.path.join(unfailed_dir, summaries_filename)

        with open(all_summaries_path, "r", encoding="utf-8") as f:
            all_summary_details = json.load(f)

        all_queries = [el["query"] for el in all_summary_details]
        all_relevant_summaries = [el["relevant_summary"] for el in all_summary_details]
        
        if os.path.exists(failed_summaries_path):
            with open(failed_summaries_path, "r", encoding="utf-8") as f:
                failed_summaries = json.load(f)
        else:
            failed_summaries = []

        failed_queries = [el["query"] for el in failed_summaries]
        failed_relevant_summaries = [el["relevant_summary"] for el in failed_summaries]

        unfailed_summaries = []
        for i, (query, summary) in enumerate(zip(all_queries, all_relevant_summaries)):
            # if summary_details not in failed_summaries:
            #     unfailed_summaries.append(summary_details)
            if query not in failed_queries and summary not in failed_relevant_summaries:
                unfailed_summaries.append(all_summary_details[i])

        with open(unfailed_summaries_path, "w", encoding="utf-8") as f:
            json.dump(unfailed_summaries, f, indent=2, ensure_ascii=False)
        break

    for el in unfailed_summaries:
        print(el["query"], el["relevant_summary"])


def main():
    MODEL_PROVIDER_LIST = [
        ("openai/gpt-5", None),
        ("anthropic/claude-sonnet-4", None),
        ("google/gemini-2.5-pro", None),
        ("moonshotai/kimi-k2-0905", "fireworks/fp8")
    ]

    model, provider = MODEL_PROVIDER_LIST[0]

    split(
        question_type="answerable",
        filter_stage="failed",
        retrieval_type="hybrid_cross-encoder",
        eval_stage="all_eval_stages",
        model=model,
        provider=provider
    )



if __name__ == "__main__":
    main()