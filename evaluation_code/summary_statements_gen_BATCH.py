# some parts copied heavily from "Evaluation of RAG Metrics for Question Answering in the Telecom Domain" paper
import os
import json
import shutil
import logging
import re
import time
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
import copy

import nltk
for resource in ["punkt", "punkt_tab"]:
    nltk.download(resource, quiet=True)
from nltk.tokenize import sent_tokenize
import numpy as np
from pydantic import BaseModel, Field

from utils.action_parsing import ActionParsingContext, get_parsed_action_by_id, get_parsed_action_as_str
from utils.exceptions import RetrievalError

load_dotenv()



### GEMINI BATCH ENDPOINT CALLING:
def get_genai_client():
    api_key = os.getenv("PAID_GEMINI_API_KEY")
    if not api_key:
        raise ValueError("PAID_GEMINI_API_KEY environment variable is required for Gemini client for evaluation generation")
    return genai.Client(
        api_key=api_key
    )


def append_to_gemini_batch_file(batch_filepath, key, prompt, response_format=None, max_reasoning_tokens=8192):
    request_body = {
        "contents": [{"parts": [{"text": prompt}]}],
    }
    if response_format is not None:
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=max_reasoning_tokens),
            response_mime_type="application/json",
            response_schema=response_format
        )
    else:
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=max_reasoning_tokens),
        )

    if hasattr(config, 'to_dict'):
        request_body["generation_config"] = config.to_dict()
    elif hasattr(config, 'to_json'):
        request_body["generation_config"] = json.loads(config.to_json())
    else:
        plain_config = {
            "thinking_config": {"thinking_budget": max_reasoning_tokens}
        }
        if response_format is not None:
            plain_config.update({
                "response_mime_type": "application/json",
                "response_schema": response_format
            })
        request_body["generation_config"] = plain_config
    
    request_obj = {
        "key": key,
        "request": request_body
    }

    if not batch_filepath.endswith(".jsonl"):
        raise ValueError("Gemini batch endpoint only supports .jsonl files currently.")
    os.makedirs(os.path.dirname(batch_filepath), exist_ok=True)
    with open(batch_filepath, 'a') as f:
        f.write(json.dumps(request_obj) + "\n")
    logging.debug(f"Wrote Gemini batch request with custom id {key} to file {batch_filepath}.")


def make_gemini_batch_request(batch_filepath, judge_model="models/gemini-2.5-pro"):
    if not batch_filepath.endswith(".jsonl"):
        raise ValueError("Gemini batch endpoint must be given .jsonl file type.")
    
    normalised_filepath = os.path.normpath(batch_filepath)
    filepath_parts = normalised_filepath.split(os.sep)
    batch_filepath_cleaned = "--".join(filepath_parts)

    client = get_genai_client()
    uploaded_file = client.files.upload(
        file=batch_filepath,
        config=types.UploadFileConfig(display_name=batch_filepath_cleaned.replace(".jsonl", ""), mime_type="jsonl")
    )
    file_batch_job = client.batches.create(
        model = judge_model,
        src = uploaded_file.name,
        config = {
            'display_name': f"file-batch-job {batch_filepath_cleaned}",
        }
    )
    logging.info(f"Created Gemini batch job: {file_batch_job.name}")
    return file_batch_job


def check_gemini_batch_status(batch_job_name):
    client = get_genai_client()
    batch_job = client.batches.get(name=batch_job_name)
    completed_states = set([
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED',
        'BATCH_STATE_COMPLETED',
        'BATCH_STATE_FAILED'
        'BATCH_STATE_CANCELLED',
        'BATCH_STATE_EXPIRED'
    ])
    # if batch_job.state.name not in completed_states:
    #     logging.info(f"Current state: {batch_job.state.name}")
    # else:
    #     logging.info(f"Job finished with state: {batch_job.state.name}")
    #     if batch_job.state.name == 'JOB_STATE_FAILED' or batch_job.state.name == 'BATCH_STATE_FAILED':
    #         logging.error(f"Error: {batch_job.error}")
    return batch_job.state


def write_gemini_batch_results(batch_job_name, output_filepath):
    success = False
    client = get_genai_client()
    batch_job = client.batches.get(name=batch_job_name)
    if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
        logging.info(f"Batch job {batch_job.name} completed successfully.")

        result_file_name = batch_job.dest.file_name
        logging.debug(f"Results are in (uploaded) file: {result_file_name}")

        logging.debug("Downloading result file content...")
        file_content = client.files.download(file=result_file_name)
        text_content = file_content.decode('utf-8')

        # Write to output file
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(text_content)
        success = True

    else:
        logging.info(f"Batch job not completed. Current state: {batch_job.state.name}")
        success = False

    return success


def check_num_open_gemini_batch_jobs():
    client = get_genai_client()
    open_states = set([
        'JOB_STATE_PENDING',
        'JOB_STATE_RUNNING',
        'JOB_STATE_QUEUED',
    ])
    batch_jobs = client.batches.list(config={"page_size": 10})
    open_jobs = [job for job in batch_jobs if job.state.name in open_states]
    num_open_jobs = len(open_jobs)
    return num_open_jobs

# check openai rate limits how many credits need to put on there.



### STATEMENT EXTRACTION.


def get_statements_prompt(question, answer):
    ## RAGAS statement splitting prompt:
    # prompt = f"""
    #     Given a question and answer, create one or more statements from each sentence in the given answer. Output the statements in the following format strictly. [Statement n]: ... where the n is the statement number.\nquestion: {question}\nanswer: {answer}\nStatements:\n
    # """.strip()

    ## Own prompt:
    prompt = f"""
Given a question and answer, create one or more statements from each sentence in the given answer. The answer will likely have references of numerical document ids especially at the end of sentences - YOU MUST NOT MENTION THESE REFERENCES OR NUMERICAL IDS IN ANY OF THE STATEMENTS YOU EXTRACT. Each statement should be fully understandable by itself, which means it needs to be self-contained. This means there should be no pronouns in the statements. Output the statements in the following format strictly. [Statement n]: ... where the n is the statement number.\nquestion: {question}\nanswer: {answer}\nStatements:\n
    """.strip()
    return prompt

    
def parse_statements_response(response):
    # Splitting statements
    response = re.sub(
        r"^.*?[0-9]+\]*[:\.]\s(.*?)", r"\1", response, flags=re.MULTILINE
    ).strip()
    statements = sent_tokenize(response)
    return statements



### ACTION DOCS PARSING AND RETRIEVAL

def get_oracle_actions(id_list, context : ActionParsingContext):
    parsed_actions = []
    for id in id_list:
        parsed_actions.append(get_parsed_action_by_id(id=id, context=context))

    return parsed_actions



### FULL METRIC EVALUATION PIPELINE


def assemble_summary_stmts(summary_obj, statements, statement_gen_model):
    question_details = {
        "query": summary_obj["query"],
        "all_relevant_qu_ids" : summary_obj["all_relevant_action_ids"],
        "regenerated_qu_ids": summary_obj["regenerated_ids"]
    }
    summary_details = {
        "summary_model": summary_obj["model"],
        "summary_provider": summary_obj["provider"],
        "relevant_summary": summary_obj["relevant_summary"],
        "summary_action_ids": summary_obj["summary_action_ids"],
        "statement_gen_model": statement_gen_model,
        "summary_statements": statements,
    }
    return {
        "question_details": question_details,
        "summary_details": summary_details
    }


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
        with open(filepath, 'r', encoding='utf-8') as f:
            summary_dicts = json.load(f)
        if not isinstance(summary_dicts, list):
            raise RetrievalError(f"Expected JSON file {filepath} to contain a list, but contained {type(summary_dicts)} instead.")
        else:
            logging.info(f"Loaded json from {filepath}, found {len(summary_dicts)} objects.")
            return summary_dicts
    except json.JSONDecodeError as e:
        raise RetrievalError(f"Error decoding JSON from file {filepath}: {str(e)}.")
    except FileNotFoundError:
        raise RetrievalError(f"File {filepath} not found.")


def write_to_json_file(data_list, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
    except TypeError as e:
        logging.error(f"Error writing to JSON file {filepath}: {str(e)}.")
        raise


def append_to_json_file(data_list, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        existing_data = []
    else:
        try:
            existing_data = read_json_file(filepath)
        except RetrievalError as e:
            raise RetrievalError(f"Error reading existing JSON from file {filepath}: {str(e)}. Cannot append to file.")
    combined_data = existing_data + data_list
    write_to_json_file(data_list=combined_data, filepath=filepath)


### BATCH REQUEST FILES CREATION

def make_summary_stmts_batch_request_file_for_file(summaries_filepath, batch_request_filepath):
    try:
        file_summary_dicts = read_json_file(summaries_filepath)
    except RetrievalError as e:
        logging.error(f"Unable to load summaries for evaluation from file {summaries_filepath}: {e}")
        return
        
    summary_dicts = copy.deepcopy(file_summary_dicts)

    requests = []

    try:
        summary_count = 0
        for current_summary_idx, summary_dict in enumerate(summary_dicts):
            if summary_dict["relevant_summary"] is None:
                logging.warning(f"Skipping summary to query {summary_dict['query']} in file {summaries_filepath} as it has None summary.")
                continue

            query = summary_dict["query"]
            summary_model = summary_dict["model"]
            summary_provider = summary_dict["provider"]
            relevant_summary = summary_dict["relevant_summary"]

            gen_summary_stmts_request_made = summary_dict.get("gen_summary_stmts_request_made", False)

            if not gen_summary_stmts_request_made:
                logging.info(f"Creating request to generate statements for summary generated by model: {summary_model} and provider: {summary_provider} to query: {query}")
                statement_gen_prompt = get_statements_prompt(question=query, answer=relevant_summary)
                requests.append((f"{parse_provider_name(summary_provider)}_{parse_model_name(summary_model)}__{query}", statement_gen_prompt))
                summary_dicts[current_summary_idx]["gen_summary_stmts_request_made"] = True
                summary_count += 1
        logging.info(f"Done creating requests for summary file {summaries_filepath}, summary_count: {summary_count}")
        # print(f"Done creating requests for summary file {summaries_filepath}, summary_count: {summary_count}")
    
    finally:
        if summary_count > 0:
            # write the new requests to batch request file
            for key, prompt in requests:
                append_to_gemini_batch_file(batch_filepath=batch_request_filepath, key=key, prompt=prompt)
            logging.info(f"Wrote {summary_count} Gemini batch requests to batch request file {batch_request_filepath}.")

            # overwrite the summaries file (it will contain the updated gen_summary_stmts_request_made field)
            write_to_json_file(data_list=summary_dicts, filepath=summaries_filepath)
            logging.info(f"Updated summaries file {summaries_filepath} gen_summary_stmts_request_made fields.")
            
        else:
            logging.info(f"No batch request file made for file {summaries_filepath}.")



def make_summary_stmts_batch_request_files_for_dir(
        qu_type="answerable",
        filter_stage="passed",
        retrieval_type="hybrid_cross-encoder",
        cleaned_summary_model_provider="_gpt-5"
    ):
    summaries_dir = os.path.join("live_summaries",f"{qu_type}_{filter_stage}_qus_summaries", retrieval_type, cleaned_summary_model_provider, "stmt_gen_annotated")
    batch_request_dir = os.path.join("batch_gen", "stmt_gen", "unrequested", f"{qu_type}_{filter_stage}_qus", retrieval_type, f"summaries_{cleaned_summary_model_provider}")
    if not os.path.exists(summaries_dir):
        logging.error(f"Summaries directory {summaries_dir} does not exist.")
        return
    else:
        logging.info(f"Starting summary statements batch request files gen of summaries in directory : {summaries_dir}")
        summaries_filenames = [name for name in sorted(os.listdir(summaries_dir)) if name.endswith(".json")]

        for summaries_filename in summaries_filenames:
            batch_request_filename = summaries_filename.replace("summaries.json", "_StmtGenRequest.jsonl")
            make_summary_stmts_batch_request_file_for_file(
                summaries_filepath = os.path.join(summaries_dir, summaries_filename),
                batch_request_filepath=os.path.join(batch_request_dir, batch_request_filename)
            )


def make_summary_stmts_batch_request_files_all():
    qu_types = ["answerable", "unanswerable"]
    filter_stages = ["passed", "failed"]
    retrieval_types = ["hybrid_cross-encoder"]
    answering_model_providers = [
        "_claude-sonnet-4",
        "_gemini-2-5-pro",
        "_gpt-5",
        "fireworks_kimi-k2-0905"
    ]
    for qu_type in qu_types:
        for filter_stage in filter_stages:
            for retrieval_type in retrieval_types:
                for answering_model_provider in answering_model_providers:
                    make_summary_stmts_batch_request_files_for_dir(
                        qu_type=qu_type,
                        filter_stage=filter_stage,
                        retrieval_type=retrieval_type,
                        cleaned_summary_model_provider=answering_model_provider
                    )



### SENDING BATCH REQUESTS

def send_batch_requests(max_batch_requests=10):
    qu_types = ["answerable", "unanswerable"]
    filter_stages = ["passed", "failed"]
    retrieval_types = ["hybrid_cross-encoder"]
    answering_model_providers = [
        "_claude-sonnet-4",
        "_gemini-2-5-pro",
        "_gpt-5",
        "fireworks_kimi-k2-0905"
    ]
    api_limit_hit = False
    max_batch_requests_hit = False
    batch_requests_made = 0
    for qu_type in qu_types:
        for filter_stage in filter_stages:
            for retrieval_type in retrieval_types:
                for answering_model_provider in answering_model_providers:
                    batch_names_for_files = []
                    unrequested_dir = os.path.join("batch_gen", "stmt_gen", "unrequested", f"{qu_type}_{filter_stage}_qus", retrieval_type, f"summaries_{answering_model_provider}")
                    requested_dir = os.path.join("batch_gen", "stmt_gen", "requested", f"{qu_type}_{filter_stage}_qus", retrieval_type, f"summaries_{answering_model_provider}")
                    for filename in os.listdir(unrequested_dir):
                        if filename.endswith(".jsonl"):
                            unrequested_batch_filepath = os.path.join(unrequested_dir, filename)
                            requested_batch_filepath = os.path.join(requested_dir, filename)
                            if batch_requests_made < max_batch_requests:
                                if check_num_open_gemini_batch_jobs() < 100:
                                    batch_job = make_gemini_batch_request(batch_filepath=unrequested_batch_filepath)
                                    logging.info(f"Sent Gemini batch request for file {unrequested_batch_filepath}, job name: {batch_job.name}")
                                    os.makedirs(os.path.dirname(requested_batch_filepath), exist_ok=True)
                                    shutil.move(unrequested_batch_filepath, os.path.join(requested_dir, filename))
                                    logging.info(f"Moved 'unrequested' batch file {filename} to 'requested' directory.")
                                    batch_names_for_files.append({
                                        "batch_filepath": requested_batch_filepath,
                                        "batch_job_name": batch_job.name
                                    })
                                else:
                                    api_limit_hit = True
                                    break
                            else:
                                max_batch_requests_hit = True
                                break
                    if len(batch_names_for_files) > 0:
                        append_to_json_file(data_list=batch_names_for_files, filepath=os.path.join(requested_dir, "batch_job_names.json"))
                    if max_batch_requests_hit:
                        logging.info(f"Reached max batch requests limit of {max_batch_requests}, stopping sending more batch requests.")
                        return
                    elif api_limit_hit:
                        logging.warning("Reached 100 open Gemini batch jobs, stopping sending more batch requests for now.")
                        return



def receive_batch_results(max_batch_checks=1000):
    qu_types = ["answerable", "unanswerable"]
    filter_stages = ["passed", "failed"]
    retrieval_types = ["hybrid_cross-encoder"]
    answering_model_providers = [
        "_claude-sonnet-4",
        "_gemini-2-5-pro",
        "_gpt-5",
        "fireworks_kimi-k2-0905"
    ]
    max_batch_checks_hit = False
    batch_checks_made = 0
    for qu_type in qu_types:
        for filter_stage in filter_stages:
            for retrieval_type in retrieval_types:
                for answering_model_provider in answering_model_providers:
                    
                    requested_batch_job_names_filepath = os.path.join("batch_gen", "stmt_gen", "requested", f"{qu_type}_{filter_stage}_qus", retrieval_type, f"summaries_{answering_model_provider}", "batch_job_names.json") 
                    if os.path.exists(requested_batch_job_names_filepath):
                        requested_batch_job_names = read_json_file(requested_batch_job_names_filepath)
                    else:
                        continue

                    for job_details in requested_batch_job_names:
                        batch_job_completed = job_details.get("batch_job_completed", False)
                        if not batch_job_completed:
                            if batch_checks_made < max_batch_checks:
                                batch_job_name = job_details["batch_job_name"]
                                batch_request_filepath = job_details["batch_filepath"]
                                output_filepath = batch_request_filepath.replace("requested", "results").replace("_StmtGenRequest.jsonl", "_StmtGenResults.jsonl")
                                success = write_gemini_batch_results(batch_job_name=batch_job_name, output_filepath=output_filepath)
                                if success:
                                    job_details["batch_job_completed"] = True
                                    logging.info(f"Wrote Gemini batch results for job {batch_request_filepath} to file {output_filepath}")
                                else:
                                    job_details["batch_job_completed"] = False
                                batch_checks_made += 1
                            else:
                                max_batch_checks_hit = True
                                break
                    
                    # overwrite the "batch_job_completed" field for each job in the json file
                    write_to_json_file(data_list=requested_batch_job_names, filepath=requested_batch_job_names_filepath)
                    if max_batch_checks_hit:
                        logging.info(f"Reached max batch checks limit of {max_batch_checks}, stopping obtaining results for more batch jobs.")
                        return
    if not max_batch_checks_hit:
        logging.info("Finished checking all batch jobs for results.")


def process_batch_results():
    qu_types = ["answerable", "unanswerable"]
    filter_stages = ["passed", "failed"]
    retrieval_types = ["hybrid_cross-encoder"]
    answering_model_providers = [
        "_claude-sonnet-4",
        "_gemini-2-5-pro",
        "_gpt-5",
        "fireworks_kimi-k2-0905"
    ]
    for qu_type in qu_types:
        for filter_stage in filter_stages:
            for retrieval_type in retrieval_types:
                for answering_model_provider in answering_model_providers:
                    pass
                    # results_dir = os.path.join("batch_gen", "stmt_gen", "results", f"{qu_type}_{filter_stage}_qus", retrieval_type, f"summaries_{answering_model_provider}")
                    # if not os.path.exists(results_dir):
                    #     continue
                    # results_filenames = [name for name in sorted(os.listdir(results_dir)) if name.endswith("_StmtGenResults.jsonl")]
                    # for results_filename in results_filenames:
                    #     results_filepath = os.path.join(results_dir, results_filename)
                    #     try:
                    #         with open(results_filepath, 'r', encoding='utf-8') as f:
                    #             lines = f.readlines()
                    #     except FileNotFoundError:
                    #         logging.error(f"Results file {results_filepath} not found.")
                    #         continue
                        
                    #     statements_generated_count = 0
                    #     summary_stmts_list = []
                    #     for line in lines:
                    #         try:
                    #             result_obj = json.loads(line)
                    #             key = result_obj.get("key")
                    #             response_parts = result_obj.get("response", {}).get("contents", [])
                    #             if response_parts and len(response_parts) > 0:
                    #                 response_text = response_parts[0].get("parts", [{}])[0].get("text", "")
                    #                 statements = parse_statements_response(response_text)
                    #                 statements_generated_count += len(statements)
                    #                 summary_stmts_list.append({
                    #                     "key": key,
                    #                     "statements": statements
                    #                 })
                    #             else:
                    #                 logging.warning(f"No response contents found for key {key} in results file {results_filepath}.")
                    #         except json.JSONDecodeError as e:
                    #             logging.error(f"Error decoding JSON line in results file {results_filepath}: {str(e)}.")
                    #             continue
                        
                    #     if statements_generated_count > 0:
                    #         output_summary_stmts_filepath = results_filepath.replace("results", "final").replace("_StmtGenResults.jsonl", "_SummaryStmts.json



### FULL PROCESS

def run_full_process():
    logging.basicConfig(level=logging.INFO, filename="logfiles/summary_statements_gen_BATCH.log", format='%(asctime)s - %(levelname)s - %(message)s')
    # disable httpx logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    make_summary_stmts_batch_request_files_all()
    logging.info("Finished making all summary statements batch request files.")
    print("Finished making all summary statements batch request files.")
    send_batch_requests()
    print("Finished sending batch requests.")
    logging.info(f"Number of open Gemini batch jobs: {check_num_open_gemini_batch_jobs()}")
    receive_batch_results()
    print("Finished receiving batch results.")




def test_batch_request():
    base_summaries_dir = "live_summaries"
    subdir = os.path.join("answerable_passed_qus_summaries","hybrid_cross-encoder","stmts_and_cited_stmts","_gpt-5")
    filename = "bg_km_AmphibianConservation_statements.json"
    with open(os.path.join(base_summaries_dir, subdir, filename), 'r', encoding='utf-8') as f:
        summary_dicts = json.load(f)
    usable = summary_dicts[2]
    used = summary_dicts[1]
    batch_filepath = os.path.join("batch_summaries_test", subdir, filename.replace(".json", "_batch.jsonl")) 
    prompt = get_statements_prompt(question=usable["question_details"]["query"], answer=usable["summary_details"]["relevant_summary"])
    # write_gemini_batch_file(batch_filepath=batch_filepath, key=usable["question_details"]["query"], prompt=prompt)
    # batch_obj = make_gemini_batch_request(batch_filepath=batch_filepath)
    batch_obj_attributes = {"name": "batches/iqzh64wx6to8mkslu7xb3js6yobego5mczs0", "display_name": "file-batch-job batch_summaries_test--answerable_passed_qus_summaries--hybrid_cross-encoder--stmts_and_cited_stmts--_gpt-5--bg_km_AmphibianConservation_statements_batch.jsonl", "state": "JOB_STATE_PENDING", "error": None, "create_time": "2025-09-29T01:25:58.946937+00:00", "start_time": None, "end_time": None, "update_time": "2025-09-29T01:25:58.946937+00:00", "model": "models/gemini-2.5-pro", "src": None, "dest": None}
    # name='batches/iqzh64wx6to8mkslu7xb3js6yobego5mczs0' display_name='file-batch-job batch_summaries_test--answerable_passed_qus_summaries--hybrid_cross-encoder--stmts_and_cited_stmts--_gpt-5--bg_km_AmphibianConservation_statements_batch.jsonl' state=<JobState.JOB_STATE_PENDING: 'JOB_STATE_PENDING'> error=None create_time=datetime.datetime(2025, 9, 29, 1, 25, 58, 946937, tzinfo=TzInfo(UTC)) start_time=None end_time=None update_time=datetime.datetime(2025, 9, 29, 1, 25, 58, 946937, tzinfo=TzInfo(UTC)) model='models/gemini-2.5-pro' src=None dest=None
    # with open(os.path.join("batch_summaries_test", "batchobjname.txt"), 'w', encoding='utf-8') as f:
        # f.write(batch_obj.name)
    checkbatch = check_gemini_batch_status(batch_job_name=batch_obj_attributes["name"])
    # print(checkbatch)
    time_taken = checkbatch.end_time - checkbatch.create_time
    # print(time_taken)

    check_num_open_gemini_batch_jobs()


def test_gemini_batch_status_all():
    qu_types = ["answerable", "unanswerable"]
    filter_stages = ["passed", "failed"]
    retrieval_types = ["hybrid_cross-encoder"]
    answering_model_providers = [
        "_claude-sonnet-4",
        "_gemini-2-5-pro",
        "_gpt-5",
        "fireworks_kimi-k2-0905"
    ]
    for qu_type in qu_types:
        for filter_stage in filter_stages:
            for retrieval_type in retrieval_types:
                for answering_model_provider in answering_model_providers:
                    requested_dir = os.path.join("batch_gen", "stmt_gen", "requested", f"{qu_type}_{filter_stage}_qus", retrieval_type, f"summaries_{answering_model_provider}")
                    batch_job_names_filepath = os.path.join(requested_dir, "batch_job_names.json")
                    try:
                        batch_jobs_for_files = read_json_file(batch_job_names_filepath)
                    except RetrievalError as e:
                        logging.error(f"Unable to load batch job names from file {batch_job_names_filepath}: {e}")
                        continue
                    
                    for job in batch_jobs_for_files:
                        batch_job_name = job["batch_job_name"]
                        batch_job_state = check_gemini_batch_status(batch_job_name=batch_job_name)
                        print(f"State: {batch_job_state}, State Name: {batch_job_state.name}, Str state: {str(batch_job_state)}")


def rescue_batch_job_names():
    client = get_genai_client()
    batch_jobs = client.batches.list(config={"page_size": 10})
    all_names = []
    for job in batch_jobs:
        display_name = job.display_name
        display_name_cleaned = display_name.replace("file-batch-job ", "")
        display_name_pathlike = display_name_cleaned.replace("--", os.sep)
        display_name_converted = display_name_pathlike.replace("unrequested", "requested")
        all_names.append({
            "batch_filepath": display_name_converted,
            "batch_job_name": job.name
        })
    
    qu_types = ["answerable", "unanswerable"]
    filter_stages = ["passed", "failed"]
    retrieval_types = ["hybrid_cross-encoder"]
    answering_model_providers = [
        "_claude-sonnet-4",
        "_gemini-2-5-pro",
        "_gpt-5",
        "fireworks_kimi-k2-0905"
    ]
    for qu_type in qu_types:
        for filter_stage in filter_stages:
            for retrieval_type in retrieval_types:
                for answering_model_provider in answering_model_providers:
                    requested_dir = os.path.join("batch_gen", "stmt_gen", "requested", f"{qu_type}_{filter_stage}_qus", retrieval_type, f"summaries_{answering_model_provider}")
                    batch_job_names_filepath = os.path.join(requested_dir, "batch_job_names.json")
                    relevant_name_objects = []
                    for name_obj in all_names:
                        if requested_dir in name_obj["batch_filepath"]:
                            relevant_name_objects.append(name_obj)
                    relevant_name_objects_sorted = sorted(relevant_name_objects, key=lambda x: x["batch_filepath"])
                    write_to_json_file(data_list=relevant_name_objects_sorted, filepath=batch_job_names_filepath)
                    os.remove(os.path.join(requested_dir, "batch_job_names_rescued.json"))



if __name__ == "__main__":
    run_full_process()

    # print(f"Number of open Gemini batch jobs: {check_num_open_gemini_batch_jobs()}")

    # job = check_gemini_batch_status(batch_job_name="batches/41cltj5ixa2kcqab5s6kgkuynfz61yw815bh")
    # print(job.state)
    # print("STRINGIFIED", str(job.state))

    # test_gemini_batch_status_all()

    # rescue_batch_job_names()


