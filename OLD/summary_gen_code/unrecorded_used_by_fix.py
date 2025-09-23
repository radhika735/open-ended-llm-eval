import os
import json
import logging

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
    
        

def sort_qus_file_for_model(qus_filepath, summaries_filepath, model, provider, counter: Counter):
    qus_dicts = read_json_file(qus_filepath)
    summaries_data = read_json_file(summaries_filepath)
    summaries_file_queries = [su["query"] for su in summaries_data]

    start_count = counter.get_count()

    for question in qus_dicts:
        query = question["question"]
        if query in summaries_file_queries:
            used_by = question.get("used_by_models", [])
            if [model, provider] not in used_by:
                counter.increment()
                logging.debug(f"Query '{query}' used by {model}, {provider} but not recorded.")
                used_by.append([model, provider])
                question["used_by_models"] = used_by

    changed_count = counter.get_count() - start_count
    if changed_count > 0:
        logging.info(f"Total queries needing updates for {model}, {provider}, {os.path.basename(qus_filepath)}: {counter.get_count()}")
        write_json_file(data_list=qus_dicts, filepath=qus_filepath)



def sort_qus_dir_for_model(model, provider, qu_type, filter_stage, counter : Counter):
    qus_dir = f"live_questions/bg_km_qus/{qu_type}/{filter_stage}/usage_annotated"
    cleaned_model_name = parse_model_name(model)
    cleaned_provider_name = parse_provider_name(provider)
    cleaned_name = f"{cleaned_provider_name}_{cleaned_model_name}"
    summaries_dir = f"summary_gen_data/{qu_type}_{filter_stage}_qus_summaries/hybrid_cross-encoder/all_eval_stages/{cleaned_name}"

    for qus_filename in sorted(os.listdir(qus_dir)):
        summary_filename = qus_filename.replace("qus","summaries")
        qus_filepath = os.path.join(qus_dir, qus_filename)
        summaries_filepath = os.path.join(summaries_dir, summary_filename)
        if os.path.exists(summaries_filepath):
            sort_qus_file_for_model(qus_filepath, summaries_filepath, model, provider, counter=counter)



def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    MODEL_PROVIDER_LIST = [
        ("openai/gpt-5", None),
        ("anthropic/claude-sonnet-4", None),
        ("google/gemini-2.5-pro", None),
        ("moonshotai/kimi-k2-0905", "fireworks/fp8")
    ]

    qu_type = "unanswerable"
    filter_stage = "failed"



if __name__ == "__main__":
    main()