# FILL IN WHICH FUNCTIONS copied heavily from "Evaluation of RAG Metrics for Question Answering in the Telecom Domain" paper
import os
import json
import shutil
import logging
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types

import nltk
for resource in ["punkt", "punkt_tab"]:
    nltk.download(resource, quiet=True)
from nltk.tokenize import sent_tokenize
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
    return batch_job.state


# Prepends new batch results to output file (file containing results of previous batches)
def prepend_gemini_batch_results(batch_job_name, output_filepath):
    batch_job_completed = False
    batch_job_successful = False
    client = get_genai_client()
    batch_job = client.batches.get(name=batch_job_name)
    completed_states = [
        'JOB_STATE_SUCCEEDED',
        'JOB_STATE_FAILED',
        'JOB_STATE_CANCELLED',
        'JOB_STATE_EXPIRED'
    ]
    if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
        logging.info(f"Batch job {batch_job.name} completed successfully.")

        result_file_name = batch_job.dest.file_name
        logging.debug(f"Results are in (uploaded) file: {result_file_name}")

        logging.debug("Downloading result file content...")
        file_content = client.files.download(file=result_file_name)
        text_content = file_content.decode('utf-8')

        # Prepend new batch result to output file
        existing_content = ""
        if os.path.exists(output_filepath):
            with open(output_filepath, 'r', encoding='utf-8') as f:
                existing_content = f.read()
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(text_content + existing_content)
        batch_job_completed = True
        batch_job_successful = True
    
    elif batch_job.state.name in completed_states:
        logging.warning(f"Batch job {batch_job.name} completed with state: {batch_job.state.name}.")
        batch_job_completed = True
        batch_job_successful = False

    else:
        logging.info(f"Batch job not completed successfully. Current state: {batch_job.state.name}")
        batch_job_completed = False
        batch_job_successful = False

    return batch_job_completed, batch_job_successful


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



### CITED STATEMENT EXTRACTION.


class CitedStatement(BaseModel):
    statement : str
    citations : list[str]


def get_citations_from_stmts_prompt(summary, statements):
    prompt = f"""
Given a summary of information and a list of statements extracted from this summary, you must extract the document IDs cited in the summary for each statement. There may be zero or more cited IDs per statement.
Output the statements with their extracted citations as a list of JSON objects.

Summary: {summary}
Statements: {statements}
    """.strip()

    return prompt


def parse_cited_stmts_response(response):
    try:
        cited_statements = json.loads(response)
    except json.JSONDecodeError as e:
        logging.error(f"During parsing cited statements from LLM response, error decoding JSON response {str(e)}. LLM response was: {response}")
        cited_statements = None
    return cited_statements



### UTILITY FUNCTIONS


def assemble_cited_stmts(statement_obj, cited_statements, cited_statements_model):
    question_details = {
        "query": statement_obj["question_details"]["query"],
        "all_relevant_qu_ids" : statement_obj["question_details"]["all_relevant_qu_ids"],
        "regenerated_qu_ids": statement_obj["regenerated_qu_ids"]
    }
    summary_details = {
        "summary_model": statement_obj["summary_details"]["summary_model"],
        "summary_provider": statement_obj["summary_details"]["summary_provider"],
        "relevant_summary": statement_obj["summary_details"]["relevant_summary"],
        "summary_action_ids": statement_obj["summary_details"]["summary_action_ids"],
        "summary_statements_model": statement_obj["summary_details"]["summary_statements_model"],
        "summary_statements": statement_obj["summary_details"]["summary_statements"],
        "cited_statements_model": cited_statements_model,
        "cited_statements": cited_statements,
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
            data = json.load(f)
        if not isinstance(data, list):
            raise RetrievalError(f"Expected JSON file {filepath} to contain a list, but contained {type(data)} instead.")
        else:
            logging.debug(f"Loaded json from {filepath}, found {len(data)} objects.")
            return data
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


def read_jsonl_file(filepath):
    data_list = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    try:
                        data = json.loads(line)
                        data_list.append(data)
                    except json.JSONDecodeError as e:
                        raise RetrievalError(f"Error decoding JSON line in file {filepath}: {str(e)}. Line content: {line}")
        return data_list
    except FileNotFoundError:
        raise RetrievalError(f"File {filepath} not found.")



### BATCH REQUEST FILES CREATION


def make_cited_stmts_batch_request_file_for_file(stmts_filepath, batch_results_filepath):
    try:
        stmts_dicts = read_json_file(stmts_filepath)
    except RetrievalError as e:
        logging.error(f"Unable to load statements from file {stmts_filepath} to create batch request file (cited statement generation): {e}")
        return
        
    requests = []

    try:
        stmts_count = 0
        for current_stmts_idx, stmts_dict in enumerate(stmts_dicts):
            if not stmts_dict["summary_details"].get("summary_statements", None):
                logging.warning(f"Skipping summary to query {stmts_dict['question_details']['query']} in file {stmts_filepath} as it does not have valid summary statements.")
                continue

            # valid (non-None and not empty) summary statements exist for this summary to query.
            
            query = stmts_dict["question_details"]["query"]
            summary_model = stmts_dict["summary_details"]["summary_model"]
            summary_provider = stmts_dict["summary_details"]["summary_provider"]
            relevant_summary = stmts_dict["summary_details"]["relevant_summary"]
            summary_stmts = stmts_dict["summary_details"]["summary_statements"]

            gen_cited_stmts_request_made = stmts_dict["summary_details"].get("gen_cited_stmts_request_made", False)

            if not gen_cited_stmts_request_made:
                logging.info(f"Creating request to generate cited statements for summary generated by model: {summary_model} and provider: {summary_provider} to query: {query}")
                cited_stmt_gen_prompt = get_citations_from_stmts_prompt(summary=relevant_summary, statements=summary_stmts)
                key = f"cited_stmt_gen__{parse_provider_name(summary_provider)}_{parse_model_name(summary_model)}__{query}"
                requests.append((key, cited_stmt_gen_prompt))
                stmts_dicts[current_stmts_idx]["summary_details"]["gen_cited_stmts_request_made"] = True
                stmts_count += 1
        
    finally:
        logging.info(f"Done creating requests for statements file {stmts_filepath}, statements_count: {stmts_count}")
    
        if stmts_count > 0:
            # write the new requests to batch request file
            cited_statements_response_schema = {
                "type": "array",
                "items" : CitedStatement.model_json_schema()
            }
            for key, prompt in requests:
                append_to_gemini_batch_file(batch_filepath=batch_results_filepath, key=key, prompt=prompt, response_format=cited_statements_response_schema)
            logging.info(f"Wrote {stmts_count} Gemini batch requests to batch request file {batch_results_filepath}.")

            # overwrite the statements file (to record the updated gen_cited_stmts_request_made fields)
            write_to_json_file(data_list=stmts_dicts, filepath=stmts_filepath)
            logging.info(f"Updated statements file {stmts_filepath} gen_cited_stmts_request_made fields.")
            
        else:
            logging.info(f"No batch request file made for file {stmts_filepath}.")


def make_cited_stmts_batch_request_files_for_dir(
        qu_type="answerable",
        filter_stage="passed",
        retrieval_type="hybrid_cross-encoder",
        cleaned_summary_model_provider="_gpt-5"
    ):
    statements_dir = os.path.join("live_summaries",f"{qu_type}_{filter_stage}_qus_summaries", retrieval_type, cleaned_summary_model_provider, "cited_stmt_gen_annotated")
    batch_request_dir = os.path.join("batch_gen", "cited_stmt_gen", "unrequested", f"{qu_type}_{filter_stage}_qus", retrieval_type, f"summaries_{cleaned_summary_model_provider}")
    if not os.path.exists(statements_dir):
        logging.error(f"Statements directory {statements_dir} does not exist.")
        return
    else:
        logging.info(f"Starting creation of batch files (requests) for cited statement generation: for summary statements in directory {statements_dir}")
        statements_filenames = [name for name in sorted(os.listdir(statements_dir)) if name.endswith(".json")]

        for statements_filename in statements_filenames:
            batch_request_filename = statements_filename.replace("statements.json", "_CitedStmtGenRequest.jsonl")
            make_cited_stmts_batch_request_file_for_file(
                stmts_filepath = os.path.join(statements_dir, statements_filename),
                batch_results_filepath=os.path.join(batch_request_dir, batch_request_filename)
            )


def make_cited_stmts_batch_request_files_all(
        qu_types = ["answerable", "unanswerable"],
        filter_stages = ["passed", "failed"],
        retrieval_types = ["hybrid_cross-encoder"],
        answering_model_providers = [
            "_claude-sonnet-4",
            "_gemini-2-5-pro",
            "_gpt-5",
            "fireworks_kimi-k2-0905"
        ]
    ):
    for qu_type in qu_types:
        for filter_stage in filter_stages:
            for retrieval_type in retrieval_types:
                for answering_model_provider in answering_model_providers:
                    make_cited_stmts_batch_request_files_for_dir(
                        qu_type=qu_type,
                        filter_stage=filter_stage,
                        retrieval_type=retrieval_type,
                        cleaned_summary_model_provider=answering_model_provider
                    )



### SENDING BATCH REQUESTS


def send_batch_requests(
        max_batch_requests=None,
        qu_types = ["answerable", "unanswerable"],
        filter_stages = ["passed", "failed"],
        retrieval_types = ["hybrid_cross-encoder"],
        answering_model_providers = [
            "_claude-sonnet-4",
            "_gemini-2-5-pro",
            "_gpt-5",
            "fireworks_kimi-k2-0905"
        ]
    ):
    api_open_requests_limit = 100
    api_limit_hit = False
    max_batch_requests_hit = False
    batch_requests_made = 0
    for qu_type in qu_types:
        for filter_stage in filter_stages:
            for retrieval_type in retrieval_types:
                for answering_model_provider in answering_model_providers:

                    unrequested_dir = os.path.join("batch_gen", "cited_stmt_gen", "unrequested", f"{qu_type}_{filter_stage}_qus", retrieval_type, f"summaries_{answering_model_provider}")
                    requested_dir = os.path.join("batch_gen", "cited_stmt_gen", "requested", f"{qu_type}_{filter_stage}_qus", retrieval_type, f"summaries_{answering_model_provider}")

                    try:
                        batch_names_for_files = read_json_file(os.path.join(requested_dir, "batch_job_names.json"))
                    except RetrievalError:
                        batch_names_for_files = []
                    
                    if not os.path.exists(unrequested_dir):
                        logging.info(f"Unrequested batch requests directory {unrequested_dir} not found, skipping sending batch requests.")
                        continue
                    for filename in os.listdir(unrequested_dir):
                        if filename.endswith(".jsonl"):
                            unrequested_batch_filepath = os.path.join(unrequested_dir, filename)
                            requested_batch_filepath = os.path.join(requested_dir, filename)

                            if max_batch_requests is None or batch_requests_made < max_batch_requests:
                                if check_num_open_gemini_batch_jobs() < api_open_requests_limit:
                                    batch_job = make_gemini_batch_request(batch_filepath=unrequested_batch_filepath)
                                    batch_requests_made += 1
                                    logging.info(f"Sent Gemini batch request for file {unrequested_batch_filepath}, job name: {batch_job.name}")
                                    os.makedirs(os.path.dirname(requested_batch_filepath), exist_ok=True)
                                    shutil.move(unrequested_batch_filepath, requested_batch_filepath)# if batch for this summaries file was previously requested (and therefore in the batch/gen/stmt_gen/requested dir), this will be overwritten with the new batchfile 
                                    logging.info(f"Moved 'unrequested' batch file {filename} to 'requested' directory.")

                                    # prepending new entry to start of list, so that if an older batch for the same summaries file exists, 
                                        # it will not be found first when searching through the list.
                                    batch_names_for_files.insert(0, {"batch_filepath": requested_batch_filepath, "batch_job_name": batch_job.name})
                                    logging.info(f"Added new batch mapping for file {requested_batch_filepath}, job name: {batch_job.name}")
                                
                                else:
                                    api_limit_hit = True
                                    break
                            else:
                                max_batch_requests_hit = True
                                break
                    if batch_requests_made > 0:
                        write_to_json_file(data_list=batch_names_for_files, filepath=os.path.join(requested_dir, "batch_job_names.json"))
                    if max_batch_requests_hit:
                        logging.info(f"Reached max batch requests limit of {max_batch_requests}, stopping sending more batch requests.")
                        return
                    elif api_limit_hit:
                        logging.warning("Reached 100 open Gemini batch jobs, stopping sending more batch requests for now.")
                        return



### RECEIVING BATCH RESULTS


def receive_batch_results(
        max_batch_checks=None,
        qu_types = ["answerable", "unanswerable"],
        filter_stages = ["passed", "failed"],
        retrieval_types = ["hybrid_cross-encoder"],
        answering_model_providers = [
            "_claude-sonnet-4",
            "_gemini-2-5-pro",
            "_gpt-5",
            "fireworks_kimi-k2-0905"
        ]
    ):
    max_batch_checks_hit = False
    batch_checks_made = 0
    for qu_type in qu_types:
        for filter_stage in filter_stages:
            for retrieval_type in retrieval_types:
                for answering_model_provider in answering_model_providers:
                    
                    requested_batch_job_names_filepath = os.path.join("batch_gen", "cited_stmt_gen", "requested", f"{qu_type}_{filter_stage}_qus", retrieval_type, f"summaries_{answering_model_provider}", "batch_job_names.json") 
                    if os.path.exists(requested_batch_job_names_filepath):
                        requested_batch_job_names = read_json_file(requested_batch_job_names_filepath)
                    else:
                        # batch job names file does not existing for this subdir,
                        #   indicating that there are no requested batch jobs for this subdir (i.e for this configuration of parameters)
                        continue

                    for job_details in requested_batch_job_names:
                        batch_job_completed = job_details.get("batch_job_completed", False)
                        if not batch_job_completed:
                            if max_batch_checks is None or batch_checks_made < max_batch_checks:
                                batch_job_name = job_details["batch_job_name"]
                                batch_request_filepath = job_details["batch_filepath"]
                                output_filepath = batch_request_filepath.replace("requested", "results").replace("_CitedStmtGenRequest.jsonl", "_CitedStmtGenResults.jsonl")
                                completed, successful = prepend_gemini_batch_results(batch_job_name=batch_job_name, output_filepath=output_filepath)
                                if completed:
                                    job_details["batch_job_completed"] = True
                                    if successful:
                                        job_details["batch_job_successful"] = True
                                        logging.info(f"Wrote Gemini batch results for job {batch_request_filepath} to file {output_filepath}")
                                    else:
                                        job_details["batch_job_successful"] = False
                                        logging.warning(f"Gemini batch job {batch_request_filepath} completed but was not successful.")
                                else:
                                    job_details["batch_job_completed"] = False
                                batch_checks_made += 1
                            else:
                                max_batch_checks_hit = True
                                break
                    
                    # overwrite the "batch_job_completed" and "batch_job_successful" fields for jobs in the batch_job_names json file
                    write_to_json_file(data_list=requested_batch_job_names, filepath=requested_batch_job_names_filepath)
                    if max_batch_checks_hit:
                        logging.info(f"Reached max batch checks limit of {max_batch_checks}, stopping obtaining results for more batch jobs.")
                        return
    if not max_batch_checks_hit:
        logging.info("Finished checking all batch jobs for results.")



### PROCESSING BATCH RESULTS (STORING GENERATED CITED STATEMENTS)


def process_batch_results_for_file(
        stmts_filepath, 
        cited_stmts_filepath, 
        batch_results_filepath,
        batch_requests_filepath, 
        batch_job_names_filepath, 
        cleaned_summary_model_provider
    ):
    try:
        stmt_dicts = read_json_file(stmts_filepath)
    except RetrievalError as e:
        logging.error(f"Unable to load statements (for recording batch results) from file {stmts_filepath}: {e}")
        return

    try:
        batch_results = read_jsonl_file(batch_results_filepath)
    except RetrievalError as e:
        logging.warning(f"Unable to load batch results from file {batch_results_filepath}: {e}")
        return
    
    try:
        batch_job_names = read_json_file(batch_job_names_filepath)
    except RetrievalError as e:
        logging.error(f"Unable to load batch job names from file {batch_job_names_filepath}: {e}")
        return

    try:
        cited_stmt_dicts = read_json_file(cited_stmts_filepath)
    except RetrievalError as e:
        logging.warning(f"Unable to load existing cited statements from file {cited_stmts_filepath}: {e}. Will create new cited statements file.")
        cited_stmt_dicts = []

    stmts_processed_count = 0
    try:
        for stmt_dict in stmt_dicts:
            if not stmt_dict["summary_details"].get("summary_statements", None):
                logging.warning(f"Skipping summary to query {stmt_dict['query']} in file {stmts_filepath} as it has invalid summary statements.")
                continue

            query = stmt_dict["question_details"]["query"]
            summary_model = stmt_dict["summary_details"]["summary_model"]
            summary_provider = stmt_dict["summary_details"]["summary_provider"]
            relevant_summary = stmt_dict["summary_details"]["relevant_summary"]

            gen_cited_stmts_received = stmt_dict["summary_details"].get("gen_cited_stmts_received", False)

            if not gen_cited_stmts_received:
                logging.info(f"Getting batch results (cited statement gen request) for statements of summary generated by model: {summary_model} and provider: {summary_provider} to query: {query}")

                # check if the (most recent) batch job corresponding to this batch_job_filepath has completed successfully:
                batch_job_exists = False
                batch_job_completed = False
                batch_job_successful = False
                for job_details in batch_job_names:
                    if job_details["batch_filepath"] == batch_requests_filepath:
                        batch_job_exists = True
                        batch_job_completed = job_details.get("batch_job_completed", False)
                        batch_job_successful = job_details.get("batch_job_successful", False)
                        break

                if not batch_job_exists:
                    logging.warning(f"No batch job found for file {batch_requests_filepath}. Cannot process batch results for statements of summary to query {query} in file {stmts_filepath}.")
                    print(f"No batch job found for file {batch_requests_filepath}. Cannot process batch results for statements of summary to query {query} in file {stmts_filepath}.")
                    continue
                elif not batch_job_completed: # batch job exists but not completed
                    logging.info(f"Skipping processing batch results for statements of summary to query {query} in file {stmts_filepath} as the corresponding batch job for file {batch_requests_filepath} has not completed yet.")
                    continue
                elif not batch_job_successful: # batch job exists and completed but was not successful
                    logging.warning(f"While processing batch results, found that batch job for file {batch_requests_filepath} completed UNSUCCESSFULLY. Resetting gen_cited_stmts_request_made and gen_cited_stmts_received flags to False for statements of summary to query {query} in file {stmts_filepath}")
                    print(f"While processing batch results, found that batch job for file {batch_requests_filepath} completed UNSUCCESSFULLY. Resetting gen_cited_stmts_request_made and gen_cited_stmts_received flags to False for statements of summary to query {query} in file {stmts_filepath}")
                    # Reset the gen_cited_stmts_request_made field to False so that cited statement generation for this summary statement list can be requested again.
                    stmt_dict["summary_details"]["gen_cited_stmts_request_made"] = False
                    stmt_dict["summary_details"]["gen_cited_stmts_received"] = False
                    continue
                
                # the batch job exists, has completed and was successful.

                # find the batch result corresponding to this summary (by key)
                batch_job_key = f"cited_stmts_gen__{cleaned_summary_model_provider}__{query}"
                batch_result_found = False
                result_cited_stmts = None
                cited_stmts_model = None
                for batch_result in batch_results:
                    if batch_result["key"] == batch_job_key:
                        batch_result_found = True
                        llm_output = batch_result["response"]["candidates"][0]["content"]["parts"][0]["text"]
                        cited_stmts_model = batch_result["response"]["modelVersion"]
                        # parse cited statements from response text
                        result_cited_stmts = parse_cited_stmts_response(response=llm_output)
                        break

                if not batch_result_found:# this condition should not happen given that the batch job exists and has completed successfully
                    logging.error(f"Batch job recorded as having completed successfully but no batch result found for key {batch_job_key} in file {batch_results_filepath}. Resetting gen_cited_stmts_request_made and gen_cited_stmts_received flags to False for this summary's statements.")
                    print(f"Batch job recorded as having completed successfully but no batch result found for key {batch_job_key} in file {batch_results_filepath}. Resetting gen_cited_stmts_request_made and gen_cited_stmts_received flags to False for this summary's statements.")
                    # Reset the gen_cited_stmts_request_made field to False so that statement generation for this summary's statements can be requested again.
                    stmt_dict["summary_details"]["gen_cited_stmts_request_made"] = False
                    stmt_dict["summary_details"]["gen_cited_stmts_received"] = False
                    continue
                elif not result_cited_stmts:
                    logging.error(f"No valid cited statements parsed from response text for key {batch_job_key} in file {batch_results_filepath}. Resetting gen_cited_stmts_request_made and gen_cited_stmts_received flags to False for this list of statements.")
                    print(f"No valid cited statements parsed from response text for key {batch_job_key} in file {batch_results_filepath}. Resetting gen_cited_stmts_request_made and gen_cited_stmts_received flags to False for this list of statements.")
                    # Reset the gen_cited_stmts_request_made field to False so that cited statement generation for this summary's statements can be requested again.
                    stmt_dict["summary_details"]["gen_cited_stmts_request_made"] = False
                    stmt_dict["summary_details"]["gen_cited_stmts_received"] = False
                    continue

                # valid cited statements parsed from llm response text in batch result object in batch results file.

                # find corresponding cited statement dict if it exists
                cited_stmt_dict_found = False
                for cited_stmt_dict in cited_stmt_dicts:
                    if cited_stmt_dict["question_details"]["query"] == query and cited_stmt_dict["summary_details"]["relevant_summary"] == relevant_summary:
                        cited_stmt_dict_found = True
                        # update it to store the generated cited statements
                        cited_stmt_dict["summary_details"]["cited_statements_model"] = cited_stmts_model
                        cited_stmt_dict["summary_details"]["cited_statements"] = result_cited_stmts
                        stmt_dict["summary_details"]["gen_cited_stmts_received"] = True
                        stmts_processed_count += 1
                        logging.info(f"Updated existing cited statements for summary's statements to query {query}. Updated in cited statements file {cited_stmts_filepath}.")
                        break
                # if corresponding cited statement dict does not exist, create a new one and append it to the list of cited_stmt_dicts
                if not cited_stmt_dict_found:
                    new_cited_stmts_dict = assemble_cited_stmts(statement_obj=stmt_dict, cited_statements=result_cited_stmts, cited_statements_model=cited_stmts_model)
                    cited_stmt_dicts.append(new_cited_stmts_dict)
                    stmt_dict["summary_details"]["gen_cited_stmts_received"] = True
                    stmts_processed_count += 1
                    logging.info(f"Appended new cited statements for statements of summary to query {query}. Appended to cited statements file {cited_stmts_filepath}.")

    finally:
        logging.info(f"Done processing batch results for statements file {stmts_filepath}")
        if stmts_processed_count > 0:
            # overwrite the cited statements file
            write_to_json_file(data_list=cited_stmt_dicts, filepath=cited_stmts_filepath)
            logging.info(f"Wrote/updated {stmts_processed_count} cited statements to cited statements file {cited_stmts_filepath}.")
        else:
            logging.info(f"No batch results processed for file {stmts_filepath}.")

        # overwrite the statements file (it will contain the updated (gen_cited_stmts_request_made and) gen_cited_stmts_received field)
        write_to_json_file(data_list=stmt_dicts, filepath=stmts_filepath)
        logging.info(f"Updated statements file {stmts_filepath} (gen_cited_stmts_request_made and) gen_cited_stmts_received fields.")

    
def process_batch_results_for_dir(qu_type, filter_stage, retrieval_type, cleaned_summary_model_provider):
    statements_dir = os.path.join("live_summaries",f"{qu_type}_{filter_stage}_qus_summaries", retrieval_type, cleaned_summary_model_provider, "cited_stmt_gen_annotated")
    cited_stmts_dir = os.path.join("live_summaries",f"{qu_type}_{filter_stage}_qus_summaries", retrieval_type, cleaned_summary_model_provider, "stmts_and_cited_stmts")
    batch_results_dir = os.path.join("batch_gen", "cited_stmt_gen", "results", f"{qu_type}_{filter_stage}_qus", retrieval_type, f"summaries_{cleaned_summary_model_provider}")
    batch_requests_dir = os.path.join("batch_gen", "cited_stmt_gen", "requested", f"{qu_type}_{filter_stage}_qus", retrieval_type, f"summaries_{cleaned_summary_model_provider}")
    if not os.path.exists(statements_dir):
        logging.error(f"Statements directory {statements_dir} does not exist.")
        return
    else:
        logging.info(f"Starting processing batch results for cited statement generation: for summary statements in directory {statements_dir}")
        statements_filenames = [name for name in sorted(os.listdir(statements_dir)) if name.endswith(".json")]

        for statements_filename in statements_filenames:
            cited_stmts_filename = statements_filename.replace("statements.json", "cited_statements.json")
            batch_results_filename = statements_filename.replace("statements.json", "_CitedStmtGenResults.jsonl")
            batch_request_filename = statements_filename.replace("statements.json", "_CitedStmtGenRequest.jsonl")
            process_batch_results_for_file(
                stmts_filepath = os.path.join(statements_dir, statements_filename),
                cited_stmts_filepath = os.path.join(cited_stmts_dir, cited_stmts_filename),
                batch_results_filepath=os.path.join(batch_results_dir, batch_results_filename),
                batch_requests_filepath=os.path.join(batch_requests_dir, batch_request_filename),
                batch_job_names_filepath=os.path.join(batch_requests_dir, "batch_job_names.json"),
                cleaned_summary_model_provider=cleaned_summary_model_provider
            )


def process_batch_results_all(
        qu_types = ["answerable", "unanswerable"],
        filter_stages = ["passed", "failed"],
        retrieval_types = ["hybrid_cross-encoder"],
        answering_model_providers = [
            "_claude-sonnet-4",
            "_gemini-2-5-pro",
            "_gpt-5",
            "fireworks_kimi-k2-0905"
        ]
    ):
    for qu_type in qu_types:
        for filter_stage in filter_stages:
            for retrieval_type in retrieval_types:
                for answering_model_provider in answering_model_providers:
                    process_batch_results_for_dir(
                        qu_type=qu_type,
                        filter_stage=filter_stage,
                        retrieval_type=retrieval_type,
                        cleaned_summary_model_provider=answering_model_provider
                    )



### CHECK IF ALL CITED STATEMENTS HAVE BEEN GENERATED AND STORED


def check_all_cited_stmts_generated_for_file(statements_filepath):
    try:
        stmt_dicts = read_json_file(statements_filepath)
    except RetrievalError as e:
        logging.error(f"Unable to load statements for checking statement generation from file {statements_filepath}: {e}")
        raise

    total_viable_stmts = 0 # viable statements are those which are a list of length greater than 0 (i.e. non-None and not empty)
    num_cited_stmts = 0
    for stmt_dict in stmt_dicts:
        if stmt_dict["summary_details"].get("summary_statements", None) is None:
            continue
        total_viable_stmts += 1
        gen_cited_stmts_received = stmt_dict["summary_details"].get("gen_cited_stmts_received", False)
        if gen_cited_stmts_received:
            num_cited_stmts += 1

    all_generated = total_viable_stmts == num_cited_stmts
    return {
        "all_generated": all_generated,
        "num_viable_statements": total_viable_stmts,
        "num_cited_stmts_generated": num_cited_stmts
    }


def check_all_cited_stmts_generated_for_dir(qu_type, filter_stage, retrieval_type, cleaned_summary_model_provider):
    statements_dir = os.path.join("live_summaries",f"{qu_type}_{filter_stage}_qus_summaries", retrieval_type, cleaned_summary_model_provider, "cited_stmt_gen_annotated")
    if not os.path.exists(statements_dir):
        raise FileNotFoundError(f"Statements directory {statements_dir} does not exist.")
    else:
        logging.debug(f"Starting checking if all cited statements have been generated: for statements in directory {statements_dir}")
        statements_filenames = [name for name in sorted(os.listdir(statements_dir)) if name.endswith(".json")]

        dir_results = {"all_generated": True, "num_viable_statements":0, "num_cited_stmts_generated":0}
        for statements_filename in statements_filenames:
            file_results = check_all_cited_stmts_generated_for_file(
                statements_filepath = os.path.join(statements_dir, statements_filename)
            )
            dir_results["all_generated"] &= file_results["all_generated"]
            dir_results["num_viable_statements"] += file_results["num_viable_statements"]
            dir_results["num_cited_stmts_generated"] += file_results["num_cited_stmts_generated"]
        return dir_results
        

def check_all_cited_stmts_generated_for_all(
        qu_types = ["answerable", "unanswerable"],
        filter_stages = ["passed", "failed"],
        retrieval_types = ["hybrid_cross-encoder"],
        answering_model_providers = [
            "_claude-sonnet-4",
            "_gemini-2-5-pro",
            "_gpt-5",
            "fireworks_kimi-k2-0905"
        ]
    ):
    all_results = {"all_generated": True, "num_viable_statements":0, "num_cited_stmts_generated":0}
    for qu_type in qu_types:
        for filter_stage in filter_stages:
            for retrieval_type in retrieval_types:
                for answering_model_provider in answering_model_providers:
                    dir_results = check_all_cited_stmts_generated_for_dir(
                        qu_type=qu_type,
                        filter_stage=filter_stage,
                        retrieval_type=retrieval_type,
                        cleaned_summary_model_provider=answering_model_provider
                    )
                    all_results["all_generated"] &= dir_results["all_generated"]
                    all_results["num_viable_statements"] += dir_results["num_viable_statements"]
                    all_results["num_cited_stmts_generated"] += dir_results["num_cited_stmts_generated"]
    return all_results



### DISPLAYING STATUS:


def print_running_jobs():
    client = get_genai_client()
    batch_jobs = client.batches.list(config={"page_size": 10})
    for job in batch_jobs:
        if job.state.name != "JOB_STATE_SUCCEEDED":
            print("Running job:", job.name, job.display_name, job.state.name)



### FULL PROCESS

def run_full_process(
        qu_types = ["answerable", "unanswerable"],
        filter_stages = ["passed", "failed"],
        retrieval_types = ["hybrid_cross-encoder"],
        summary_model_providers = [
            "_claude-sonnet-4",
            "_gemini-2-5-pro",
            "_gpt-5",
            "fireworks_kimi-k2-0905"
        ],
        max_batch_requests=None,
        max_batch_retrievals=None
    ):
    logging.basicConfig(level=logging.INFO, filename="logfiles/cited_summary_statements_gen_batch.log", format='%(asctime)s - %(levelname)s - %(message)s')
    # disable httpx logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    logging.info(f"STARTING FULL PROCESS FOR CITED SUMMARY STATEMENT GENERATION USING GEMINI BATCH API. There are currently {check_num_open_gemini_batch_jobs()} open Gemini batch jobs.")

    logging.info("Starting making batch request files for cited summary statement generation, for all summary statements.")
    make_cited_stmts_batch_request_files_all(
        qu_types=qu_types, 
        filter_stages=filter_stages, 
        retrieval_types=retrieval_types, 
        answering_model_providers=summary_model_providers
    )
    logging.info("Finished making batch request files for cited statement generation, for all summary statements.")
    print("Finished making batch request files for cited statement generation, for all summary statements.")

    logging.info("Starting sending batch requests.")
    send_batch_requests(
        max_batch_requests=max_batch_requests,
        qu_types=qu_types, 
        filter_stages=filter_stages, 
        retrieval_types=retrieval_types, 
        answering_model_providers=summary_model_providers
    )
    logging.info("Finished sending batch requests.")
    print("Finished sending batch requests.")
    logging.info(f"Number of open Gemini batch jobs: {check_num_open_gemini_batch_jobs()}")

    logging.info("Starting receiving batch results.")
    receive_batch_results(
        max_batch_checks=max_batch_retrievals,
        qu_types=qu_types,
        filter_stages=filter_stages,
        retrieval_types=retrieval_types,
        answering_model_providers=summary_model_providers
    )
    logging.info("Finished receiving batch results.")
    print("Finished receiving batch results.")

    logging.info("Starting processing batch results for cited statement generation, for all summary statements.")
    process_batch_results_all(
        qu_types=qu_types,
        filter_stages=filter_stages,
        retrieval_types=retrieval_types,
        answering_model_providers=summary_model_providers
    )
    logging.info("Finished processing batch results for cited statement generation, for all summary statements.")
    print("Finished processing batch results for cited statement generation, for all summary statements.")

    logging.info(f"There are now {check_num_open_gemini_batch_jobs()} open Gemini batch jobs.")
    stmts_generation_details = check_all_cited_stmts_generated_for_all(
        qu_types=qu_types,
        filter_stages=filter_stages,
        retrieval_types=retrieval_types,
        answering_model_providers=summary_model_providers
    )
    if stmts_generation_details["all_generated"]:
        logging.info(f"All cited statements have been generated and stored. Number of viable statements: {stmts_generation_details['num_viable_statements']}, number of summaries with generated cited statements: {stmts_generation_details['num_cited_stmts_generated']}.")
        print(f"All cited statements have been generated and stored. Number of viable statements: {stmts_generation_details['num_viable_statements']}, number of summaries with generated cited statements: {stmts_generation_details['num_cited_stmts_generated']}.")
    else:
        logging.info(f"Not all cited statements have been generated and stored. Run the program again to generate and record more cited statements. Number of viable statements: {stmts_generation_details['num_viable_statements']}, number of summaries with generated cited statements: {stmts_generation_details['num_cited_stmts_generated']}.")
        print(f"Not all cited statements have been generated and stored. Run the program again to generate and record more cited statements. Number of viable statements: {stmts_generation_details['num_viable_statements']}, number of summaries with generated cited statements: {stmts_generation_details['num_cited_stmts_generated']}.")
    logging.info(f"ENDING PROCESS FOR CITED STATEMENT GENERATION USING GEMINI BATCH API." )


def test_inline_batch_request():
    query = "How does the success of providing artificial nest sites for solitary bees compare to providing them for bumblebees?"
    relevant_summary = "The success of providing artificial nest sites differs significantly between solitary bees and bumblebees, with solitary bees showing much higher success rates. For solitary bees, 29 out of 30 replicated trials showed occupancy by bees across 10 countries including Europe, North and South America and Asia (Action 47). Occupancy rates for solitary bee nest boxes, where reported, were between 1 and 26% of available cavities (Action 80). The number of occupied solitary bee nests can double over three years with repeated nest box provision at a given site, as demonstrated in three trials on agricultural land (Action 47). One study estimated that the number of foraging Osmia bees had increased in fields with nest boxes compared to fields without nest boxes (Action 47). In contrast, bumblebee nest box success is much lower. Three replicated trials since 1989 in the UK showed very low uptake rates of 0-2.5% for various nest box designs, while seven earlier trials in the UK, USA or Canada showed overall uptake rates between 10% and 48% (Action 48). Two replicated trials tested bumblebee nest boxes and both found very low uptake of 2% or less (Action 80). Underground nest boxes buried 5-10 cm underground with entrance pipes are generally the most effective for bumblebees, with seven replicated trials finding between 6% and 58% occupancy, but even these rates are often lower than solitary bee success rates (Action 48). However, there is no evidence captured for the effects of providing nest boxes on bumblebee populations (Action 48).",
    statements = [
        "The success of providing artificial nest sites differs significantly between solitary bees and bumblebees.",
        "Solitary bees show much higher success rates in occupying artificial nest sites than bumblebees do.",
        "For solitary bees, 29 out of 30 replicated trials showed occupancy by bees across 10 countries, including Europe, North and South America, and Asia.",
        "Reported occupancy rates for solitary bee nest boxes were between 1% and 26% of available cavities.",
        "The number of occupied solitary bee nests can double over three years with repeated nest box provision at a given site.",
        "The doubling of occupied solitary bee nests was demonstrated in three trials on agricultural land.",
        "One study estimated that the number of foraging Osmia bees had increased in fields with nest boxes compared to fields without nest boxes.",
        "In contrast to solitary bees, bumblebee nest box success is much lower.",
        "Three replicated trials since 1989 in the UK showed very low uptake rates of 0-2.5% for various bumblebee nest box designs.",
        "Seven earlier trials in the UK, USA, or Canada showed overall bumblebee nest box uptake rates between 10% and 48%.",
        "Two replicated trials that tested bumblebee nest boxes both found very low uptake of 2% or less.",
        "Underground nest boxes buried 5-10 cm underground with entrance pipes are generally the most effective for bumblebees.",
        "Seven replicated trials of underground bumblebee nest boxes found between 6% and 58% occupancy.",
        "The occupancy rates for even the most effective bumblebee nest boxes are often lower than solitary bee success rates.",
        "There is no evidence captured for the effects of providing nest boxes on bumblebee populations."
    ]

    query2 = "What actions related to reducing farming intensity can be taken to benefit wild bees?"
    relevant_summary2 = "Several actions related to reducing farming intensity can benefit wild bees. Four replicated trials in Europe have compared farmland meadows managed extensively with conventionally farmed meadows, with two trials finding enhanced numbers and diversity of wild bees on meadows with a delayed first cut and little agrochemical use (Action 22). Reducing grazing intensity on pastures can increase the abundance of cavity-nesting bees and wasps, as shown by one replicated trial demonstrating that reducing summer cattle grazing intensity increased bee abundance (Action 23). \n\nEvidence on reduced tillage shows mixed results - two replicated trials on squash farms in the USA had contrasting outcomes, with one showing no difference in bee abundance between tilled and untilled farms, while another found three times more squash bees on no-till farms than on conventional farms (Action 11).\n\nConverting to organic farming shows equivocal evidence - three replicated trials in Europe or Canada demonstrated higher wild bee abundance under organic arable farming than conventional farming, with one trial showing higher bee diversity in organically farmed wheat fields. However, three other replicated trials in Europe or the USA found no significant difference in bumblebee numbers, bumblebee species, or wild bees visiting flowering crops between conventional and organic farms (Action 25).\n\nRegarding pesticide reduction, one replicated trial in the USA showed that numbers of foraging bees on squash farms are not affected by the responsible use of pesticides (Action 27). Agri-environment schemes have shown variable results - four replicated trials in Europe demonstrated enhanced diversity and/or abundance of foraging wild bees under various European agri-environment schemes compared to conventionally-managed fields, while four other replicated trials found no enhancement in bee numbers or species under agri-environment schemes (Action 24).",
    statements2 = [
        "Several actions related to reducing farming intensity can benefit wild bees.",
        "Two replicated trials in Europe found enhanced numbers and diversity of wild bees on farmland meadows with a delayed first cut and little agrochemical use compared to conventionally farmed meadows.",
        "Reducing grazing intensity on pastures can increase the abundance of cavity-nesting bees and wasps.",
        "One replicated trial demonstrated that reducing summer cattle grazing intensity increased bee abundance.",
        "Evidence on the effect of reduced tillage on bees shows mixed results.",
        "One replicated trial on squash farms in the USA showed no difference in bee abundance between tilled and untilled farms.",
        "Another replicated trial on squash farms in the USA found three times more squash bees on no-till farms than on conventional farms.",
        "Converting to organic farming shows equivocal evidence for benefitting wild bees.",
        "Three replicated trials in Europe or Canada demonstrated higher wild bee abundance under organic arable farming than under conventional farming.",
        "One trial showed higher bee diversity in organically farmed wheat fields.",
        "Three other replicated trials in Europe or the USA found no significant difference in bumblebee numbers, bumblebee species, or wild bees visiting flowering crops between conventional and organic farms.",
        "One replicated trial in the USA showed that numbers of foraging bees on squash farms are not affected by the responsible use of pesticides.",
        "Agri-environment schemes have shown variable results for wild bees.",
        "Four replicated trials in Europe demonstrated enhanced diversity and/or abundance of foraging wild bees under various European agri-environment schemes compared to conventionally-managed fields.",
        "Four other replicated trials found no enhancement in bee numbers or species under agri-environment schemes."
    ]


    batch_filepath = os.path.join("batch_cited_stmt_gen_test", "request_file7.jsonl")
    # cited_statements_response_schema = {
    #     "type": "array",
    #     "items" : CitedStatement.model_json_schema()
    # }
    # cited_statements_response_format = {
    #     "type": "json_schema",
    #     "json_schema": {
    #         "name" : "ListOfCitedStatements",
    #         "strict" : True, 
    #         "schema": cited_statements_response_schema
    #     }
    # }
    prompt = get_citations_from_stmts_prompt(summary=relevant_summary, statements=statements)
    key = f"TEST__cited_stmt_gen___claude-sonnet-4__{query}"
    append_to_gemini_batch_file(batch_filepath=batch_filepath, prompt=prompt, key=key)
    prompt2 = get_citations_from_stmts_prompt(summary=relevant_summary2, statements=statements2)
    key2 = f"TEST__cited_stmt_gen___claude-sonnet-4__{query2}"
    append_to_gemini_batch_file(batch_filepath=batch_filepath, prompt=prompt2, key=key2)
    # batch_job = make_gemini_batch_request(batch_filepath=batch_filepath)

    with open(batch_filepath, 'r', encoding='utf-8') as f:
        batch_request = [json.loads(line) for line in f if line.strip()]

    client = get_genai_client()
    inline_requests = []
    for req in batch_request:
        inline_requests.append({
            'contents': [{
                'parts': [{'text': req["request"]["contents"][0]["parts"][0]["text"]}],
            }],
            'config': {
                'response_mime_type': 'application/json',
                'response_schema': list[CitedStatement],
            }
        })
    
    inline_batch_job = client.batches.create(
        model="gemini-2.5-pro",
        src = inline_requests,
        config={
            'display_name': "batch_cited_stmt_gen_test7"
        },
    )

    print(inline_batch_job)
    with open(os.path.join("batch_cited_stmt_gen_test", "batchjobname7.txt"), 'w', encoding='utf-8') as f:
        f.write(inline_batch_job.name)

    
def test_inline_batch_response():
    client = get_genai_client()
    with open(os.path.join("batch_cited_stmt_gen_test", "batchjobname7.txt"), 'r', encoding='utf-8') as f:
        job_name = f.read().strip()
    batch_job_inline = client.batches.get(name=job_name)
    if batch_job_inline.state.name in ('JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'):
        print("Job completed")
        data = []
        for i, inline_response in enumerate(batch_job_inline.dest.inlined_responses, start=1):
            print(f"Response {i+1}:")
            if inline_response.response:
                # Accessing response, structure may vary.
                try:
                    print(inline_response.response.text)
                    data.append(json.loads(inline_response.response.text))
                except AttributeError:
                    print(inline_response.response) # Fallback
                    data.append(json.loads(inline_response.response))
            elif inline_response.error:
                print(f"Error: {inline_response.error}")
        with open(os.path.join("batch_cited_stmt_gen_test", "batch_results7.json"), 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        print("Job not completed:", batch_job_inline.state.name)


def analyse_config():
    x = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=8000),
        response_mime_type="application/json",
        response_schema=list[CitedStatement]
    )
    print(x)
    # response_schema=list[__main__.CitedStatement]


def test_file_batch_request():
    query = "How does the success of providing artificial nest sites for solitary bees compare to providing them for bumblebees?"
    relevant_summary = "The success of providing artificial nest sites differs significantly between solitary bees and bumblebees, with solitary bees showing much higher success rates. For solitary bees, 29 out of 30 replicated trials showed occupancy by bees across 10 countries including Europe, North and South America and Asia (Action 47). Occupancy rates for solitary bee nest boxes, where reported, were between 1 and 26% of available cavities (Action 80). The number of occupied solitary bee nests can double over three years with repeated nest box provision at a given site, as demonstrated in three trials on agricultural land (Action 47). One study estimated that the number of foraging Osmia bees had increased in fields with nest boxes compared to fields without nest boxes (Action 47). In contrast, bumblebee nest box success is much lower. Three replicated trials since 1989 in the UK showed very low uptake rates of 0-2.5% for various nest box designs, while seven earlier trials in the UK, USA or Canada showed overall uptake rates between 10% and 48% (Action 48). Two replicated trials tested bumblebee nest boxes and both found very low uptake of 2% or less (Action 80). Underground nest boxes buried 5-10 cm underground with entrance pipes are generally the most effective for bumblebees, with seven replicated trials finding between 6% and 58% occupancy, but even these rates are often lower than solitary bee success rates (Action 48). However, there is no evidence captured for the effects of providing nest boxes on bumblebee populations (Action 48).",
    statements = [
        "The success of providing artificial nest sites differs significantly between solitary bees and bumblebees.",
        "Solitary bees show much higher success rates in occupying artificial nest sites than bumblebees do.",
        "For solitary bees, 29 out of 30 replicated trials showed occupancy by bees across 10 countries, including Europe, North and South America, and Asia.",
        "Reported occupancy rates for solitary bee nest boxes were between 1% and 26% of available cavities.",
        "The number of occupied solitary bee nests can double over three years with repeated nest box provision at a given site.",
        "The doubling of occupied solitary bee nests was demonstrated in three trials on agricultural land.",
        "One study estimated that the number of foraging Osmia bees had increased in fields with nest boxes compared to fields without nest boxes.",
        "In contrast to solitary bees, bumblebee nest box success is much lower.",
        "Three replicated trials since 1989 in the UK showed very low uptake rates of 0-2.5% for various bumblebee nest box designs.",
        "Seven earlier trials in the UK, USA, or Canada showed overall bumblebee nest box uptake rates between 10% and 48%.",
        "Two replicated trials that tested bumblebee nest boxes both found very low uptake of 2% or less.",
        "Underground nest boxes buried 5-10 cm underground with entrance pipes are generally the most effective for bumblebees.",
        "Seven replicated trials of underground bumblebee nest boxes found between 6% and 58% occupancy.",
        "The occupancy rates for even the most effective bumblebee nest boxes are often lower than solitary bee success rates.",
        "There is no evidence captured for the effects of providing nest boxes on bumblebee populations."
    ]

    query2 = "What actions related to reducing farming intensity can be taken to benefit wild bees?"
    relevant_summary2 = "Several actions related to reducing farming intensity can benefit wild bees. Four replicated trials in Europe have compared farmland meadows managed extensively with conventionally farmed meadows, with two trials finding enhanced numbers and diversity of wild bees on meadows with a delayed first cut and little agrochemical use (Action 22). Reducing grazing intensity on pastures can increase the abundance of cavity-nesting bees and wasps, as shown by one replicated trial demonstrating that reducing summer cattle grazing intensity increased bee abundance (Action 23). \n\nEvidence on reduced tillage shows mixed results - two replicated trials on squash farms in the USA had contrasting outcomes, with one showing no difference in bee abundance between tilled and untilled farms, while another found three times more squash bees on no-till farms than on conventional farms (Action 11).\n\nConverting to organic farming shows equivocal evidence - three replicated trials in Europe or Canada demonstrated higher wild bee abundance under organic arable farming than conventional farming, with one trial showing higher bee diversity in organically farmed wheat fields. However, three other replicated trials in Europe or the USA found no significant difference in bumblebee numbers, bumblebee species, or wild bees visiting flowering crops between conventional and organic farms (Action 25).\n\nRegarding pesticide reduction, one replicated trial in the USA showed that numbers of foraging bees on squash farms are not affected by the responsible use of pesticides (Action 27). Agri-environment schemes have shown variable results - four replicated trials in Europe demonstrated enhanced diversity and/or abundance of foraging wild bees under various European agri-environment schemes compared to conventionally-managed fields, while four other replicated trials found no enhancement in bee numbers or species under agri-environment schemes (Action 24).",
    statements2 = [
        "Several actions related to reducing farming intensity can benefit wild bees.",
        "Two replicated trials in Europe found enhanced numbers and diversity of wild bees on farmland meadows with a delayed first cut and little agrochemical use compared to conventionally farmed meadows.",
        "Reducing grazing intensity on pastures can increase the abundance of cavity-nesting bees and wasps.",
        "One replicated trial demonstrated that reducing summer cattle grazing intensity increased bee abundance.",
        "Evidence on the effect of reduced tillage on bees shows mixed results.",
        "One replicated trial on squash farms in the USA showed no difference in bee abundance between tilled and untilled farms.",
        "Another replicated trial on squash farms in the USA found three times more squash bees on no-till farms than on conventional farms.",
        "Converting to organic farming shows equivocal evidence for benefitting wild bees.",
        "Three replicated trials in Europe or Canada demonstrated higher wild bee abundance under organic arable farming than under conventional farming.",
        "One trial showed higher bee diversity in organically farmed wheat fields.",
        "Three other replicated trials in Europe or the USA found no significant difference in bumblebee numbers, bumblebee species, or wild bees visiting flowering crops between conventional and organic farms.",
        "One replicated trial in the USA showed that numbers of foraging bees on squash farms are not affected by the responsible use of pesticides.",
        "Agri-environment schemes have shown variable results for wild bees.",
        "Four replicated trials in Europe demonstrated enhanced diversity and/or abundance of foraging wild bees under various European agri-environment schemes compared to conventionally-managed fields.",
        "Four other replicated trials found no enhancement in bee numbers or species under agri-environment schemes."
    ]

    num = "8"

    batch_filepath = os.path.join("batch_cited_stmt_gen_test", f"request_file{num}.jsonl")
    cited_statements_response_schema = {
        "type": "array",
        "items" : CitedStatement.model_json_schema()
    }
    cited_statements_response_format = {
        "type": "json_schema",
        "json_schema": {
            "name" : "ListOfCitedStatements",
            "strict" : True, 
            "schema": cited_statements_response_schema
        }
    }

    my_list_schema_dict = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "statement": {"type": "string"},
                "citations": {
                    "type": "array",
                    "items": {"type": "string"}
                },
            },
            "required": ["statement", "citations"],
            "additionalProperties": False,
        }
    }

    prompt = get_citations_from_stmts_prompt(summary=relevant_summary, statements=statements)
    key = f"TEST__cited_stmt_gen___claude-sonnet-4__{query}"
    append_to_gemini_batch_file(batch_filepath=batch_filepath, prompt=prompt, key=key, response_format=my_list_schema_dict)
    prompt2 = get_citations_from_stmts_prompt(summary=relevant_summary2, statements=statements2)
    key2 = f"TEST__cited_stmt_gen___claude-sonnet-4__{query2}"
    append_to_gemini_batch_file(batch_filepath=batch_filepath, prompt=prompt2, key=key2, response_format=my_list_schema_dict)
    batch_job = make_gemini_batch_request(batch_filepath=batch_filepath)

    with open(os.path.join("batch_cited_stmt_gen_test", f"batchjobname{num}.txt"), 'w', encoding='utf-8') as f:
        f.write(batch_job.name)


def test_file_batch_response():
    num = "8"
    client = get_genai_client()
    with open(os.path.join("batch_cited_stmt_gen_test", f"batchjobname{num}.txt"), 'r', encoding='utf-8') as f:
        job_name = f.read().strip()
    batch_job = client.batches.get(name=job_name)

    if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
        print(f"Batch job {batch_job.name} completed successfully.")

        result_file_name = batch_job.dest.file_name

        file_content = client.files.download(file=result_file_name)
        text_content = file_content.decode('utf-8')

        # Prepend new batch result to output file
        output_filepath = os.path.join("batch_cited_stmt_gen_test", f"batch_results{num}.jsonl")
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(text_content)
    else:
        print("Batch job not completed:", batch_job.state.name)


def examine_schema():
    cited_statements_response_schema = {
        "type": "array",
        "items" : CitedStatement.model_json_schema()
    }
    cited_statements_response_format = {
        "type": "json_schema",
        "json_schema": {
            "name" : "ListOfCitedStatements",
            "strict" : True, 
            "schema": cited_statements_response_schema
        }
    }

    final_schema = {
        "$defs":{
            "Schema":{
                "additionalProperties": False,
                "description": "List Of Cited Statements",
                "properties": {
                    "type": "array",
                    "items": {
                        "$ref": "#/$defs/CitedStatement"
                    }

                },
                "title": "Schema",
                "type": "object"
            }
        }
    }


    print(cited_statements_response_schema)



def test_structured_batch_file_request_1():
    num = "1"
    query = "How does the success of providing artificial nest sites for solitary bees compare to providing them for bumblebees?"
    relevant_summary = "The success of providing artificial nest sites differs significantly between solitary bees and bumblebees, with solitary bees showing much higher success rates. For solitary bees, 29 out of 30 replicated trials showed occupancy by bees across 10 countries including Europe, North and South America and Asia (Action 47). Occupancy rates for solitary bee nest boxes, where reported, were between 1 and 26% of available cavities (Action 80). The number of occupied solitary bee nests can double over three years with repeated nest box provision at a given site, as demonstrated in three trials on agricultural land (Action 47). One study estimated that the number of foraging Osmia bees had increased in fields with nest boxes compared to fields without nest boxes (Action 47). In contrast, bumblebee nest box success is much lower. Three replicated trials since 1989 in the UK showed very low uptake rates of 0-2.5% for various nest box designs, while seven earlier trials in the UK, USA or Canada showed overall uptake rates between 10% and 48% (Action 48). Two replicated trials tested bumblebee nest boxes and both found very low uptake of 2% or less (Action 80). Underground nest boxes buried 5-10 cm underground with entrance pipes are generally the most effective for bumblebees, with seven replicated trials finding between 6% and 58% occupancy, but even these rates are often lower than solitary bee success rates (Action 48). However, there is no evidence captured for the effects of providing nest boxes on bumblebee populations (Action 48).",
    statements = [
        "The success of providing artificial nest sites differs significantly between solitary bees and bumblebees.",
        "Solitary bees show much higher success rates in occupying artificial nest sites than bumblebees do.",
        "For solitary bees, 29 out of 30 replicated trials showed occupancy by bees across 10 countries, including Europe, North and South America, and Asia.",
        "Reported occupancy rates for solitary bee nest boxes were between 1% and 26% of available cavities.",
        "The number of occupied solitary bee nests can double over three years with repeated nest box provision at a given site.",
        "The doubling of occupied solitary bee nests was demonstrated in three trials on agricultural land.",
        "One study estimated that the number of foraging Osmia bees had increased in fields with nest boxes compared to fields without nest boxes.",
        "In contrast to solitary bees, bumblebee nest box success is much lower.",
        "Three replicated trials since 1989 in the UK showed very low uptake rates of 0-2.5% for various bumblebee nest box designs.",
        "Seven earlier trials in the UK, USA, or Canada showed overall bumblebee nest box uptake rates between 10% and 48%.",
        "Two replicated trials that tested bumblebee nest boxes both found very low uptake of 2% or less.",
        "Underground nest boxes buried 5-10 cm underground with entrance pipes are generally the most effective for bumblebees.",
        "Seven replicated trials of underground bumblebee nest boxes found between 6% and 58% occupancy.",
        "The occupancy rates for even the most effective bumblebee nest boxes are often lower than solitary bee success rates.",
        "There is no evidence captured for the effects of providing nest boxes on bumblebee populations."
    ]

    cited_statements_format = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "statement": {"type": "STRING"},
                "citations": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                },
            },
            "required": ["statement", "citations"],
            "additionalProperties": False,
        }
    }



def test_simple_inline_request():
    prompt = "List five fruits."
    client = get_genai_client()
    inline_requests = [{
        'contents': [{
            'parts': [{'text': prompt}],
        }],

    }]
    
    inline_batch_job = client.batches.create(
        model="gemini-2.5-pro",
        src = inline_requests,
        config={
            'display_name': "simple_inline_test"
        },
    )

    print(inline_batch_job)
    with open(os.path.join("batch_cited_stmt_gen_test", "simple_inline_batchjobname.txt"), 'w', encoding='utf-8') as f:
        f.write(inline_batch_job.name)



def test_structured_new_batch_file_request_1():
    num = "1"

    query = "How does the success of providing artificial nest sites for solitary bees compare to providing them for bumblebees?"
    relevant_summary = "The success of providing artificial nest sites differs significantly between solitary bees and bumblebees, with solitary bees showing much higher success rates. For solitary bees, 29 out of 30 replicated trials showed occupancy by bees across 10 countries including Europe, North and South America and Asia (Action 47). Occupancy rates for solitary bee nest boxes, where reported, were between 1 and 26% of available cavities (Action 80). The number of occupied solitary bee nests can double over three years with repeated nest box provision at a given site, as demonstrated in three trials on agricultural land (Action 47). One study estimated that the number of foraging Osmia bees had increased in fields with nest boxes compared to fields without nest boxes (Action 47). In contrast, bumblebee nest box success is much lower. Three replicated trials since 1989 in the UK showed very low uptake rates of 0-2.5% for various nest box designs, while seven earlier trials in the UK, USA or Canada showed overall uptake rates between 10% and 48% (Action 48). Two replicated trials tested bumblebee nest boxes and both found very low uptake of 2% or less (Action 80). Underground nest boxes buried 5-10 cm underground with entrance pipes are generally the most effective for bumblebees, with seven replicated trials finding between 6% and 58% occupancy, but even these rates are often lower than solitary bee success rates (Action 48). However, there is no evidence captured for the effects of providing nest boxes on bumblebee populations (Action 48).",
    statements = [
        "The success of providing artificial nest sites differs significantly between solitary bees and bumblebees.",
        "Solitary bees show much higher success rates in occupying artificial nest sites than bumblebees do.",
        "For solitary bees, 29 out of 30 replicated trials showed occupancy by bees across 10 countries, including Europe, North and South America, and Asia.",
        "Reported occupancy rates for solitary bee nest boxes were between 1% and 26% of available cavities.",
        "The number of occupied solitary bee nests can double over three years with repeated nest box provision at a given site.",
        "The doubling of occupied solitary bee nests was demonstrated in three trials on agricultural land.",
        "One study estimated that the number of foraging Osmia bees had increased in fields with nest boxes compared to fields without nest boxes.",
        "In contrast to solitary bees, bumblebee nest box success is much lower.",
        "Three replicated trials since 1989 in the UK showed very low uptake rates of 0-2.5% for various bumblebee nest box designs.",
        "Seven earlier trials in the UK, USA, or Canada showed overall bumblebee nest box uptake rates between 10% and 48%.",
        "Two replicated trials that tested bumblebee nest boxes both found very low uptake of 2% or less.",
        "Underground nest boxes buried 5-10 cm underground with entrance pipes are generally the most effective for bumblebees.",
        "Seven replicated trials of underground bumblebee nest boxes found between 6% and 58% occupancy.",
        "The occupancy rates for even the most effective bumblebee nest boxes are often lower than solitary bee success rates.",
        "There is no evidence captured for the effects of providing nest boxes on bumblebee populations."
    ]

    query2 = "What actions related to reducing farming intensity can be taken to benefit wild bees?"
    relevant_summary2 = "Several actions related to reducing farming intensity can benefit wild bees. Four replicated trials in Europe have compared farmland meadows managed extensively with conventionally farmed meadows, with two trials finding enhanced numbers and diversity of wild bees on meadows with a delayed first cut and little agrochemical use (Action 22). Reducing grazing intensity on pastures can increase the abundance of cavity-nesting bees and wasps, as shown by one replicated trial demonstrating that reducing summer cattle grazing intensity increased bee abundance (Action 23). \n\nEvidence on reduced tillage shows mixed results - two replicated trials on squash farms in the USA had contrasting outcomes, with one showing no difference in bee abundance between tilled and untilled farms, while another found three times more squash bees on no-till farms than on conventional farms (Action 11).\n\nConverting to organic farming shows equivocal evidence - three replicated trials in Europe or Canada demonstrated higher wild bee abundance under organic arable farming than conventional farming, with one trial showing higher bee diversity in organically farmed wheat fields. However, three other replicated trials in Europe or the USA found no significant difference in bumblebee numbers, bumblebee species, or wild bees visiting flowering crops between conventional and organic farms (Action 25).\n\nRegarding pesticide reduction, one replicated trial in the USA showed that numbers of foraging bees on squash farms are not affected by the responsible use of pesticides (Action 27). Agri-environment schemes have shown variable results - four replicated trials in Europe demonstrated enhanced diversity and/or abundance of foraging wild bees under various European agri-environment schemes compared to conventionally-managed fields, while four other replicated trials found no enhancement in bee numbers or species under agri-environment schemes (Action 24).",
    statements2 = [
        "Several actions related to reducing farming intensity can benefit wild bees.",
        "Two replicated trials in Europe found enhanced numbers and diversity of wild bees on farmland meadows with a delayed first cut and little agrochemical use compared to conventionally farmed meadows.",
        "Reducing grazing intensity on pastures can increase the abundance of cavity-nesting bees and wasps.",
        "One replicated trial demonstrated that reducing summer cattle grazing intensity increased bee abundance.",
        "Evidence on the effect of reduced tillage on bees shows mixed results.",
        "One replicated trial on squash farms in the USA showed no difference in bee abundance between tilled and untilled farms.",
        "Another replicated trial on squash farms in the USA found three times more squash bees on no-till farms than on conventional farms.",
        "Converting to organic farming shows equivocal evidence for benefitting wild bees.",
        "Three replicated trials in Europe or Canada demonstrated higher wild bee abundance under organic arable farming than under conventional farming.",
        "One trial showed higher bee diversity in organically farmed wheat fields.",
        "Three other replicated trials in Europe or the USA found no significant difference in bumblebee numbers, bumblebee species, or wild bees visiting flowering crops between conventional and organic farms.",
        "One replicated trial in the USA showed that numbers of foraging bees on squash farms are not affected by the responsible use of pesticides.",
        "Agri-environment schemes have shown variable results for wild bees.",
        "Four replicated trials in Europe demonstrated enhanced diversity and/or abundance of foraging wild bees under various European agri-environment schemes compared to conventionally-managed fields.",
        "Four other replicated trials found no enhancement in bee numbers or species under agri-environment schemes."
    ]

    batch_filepath = os.path.join("batch_cited_stmt_gen_test", f"structured_new_request_file{num}.jsonl")
    cited_statements_format = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "statement": {"type": "STRING"},
                "citations": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                },
            },
            "required": ["statement", "citations"],
            "additionalProperties": False,
        }
    }

    prompt = get_citations_from_stmts_prompt(summary=relevant_summary, statements=statements)
    key = f"TEST_structured_new__cited_stmt_gen___claude-sonnet-4__{query}"
    append_to_gemini_batch_file(batch_filepath=batch_filepath, prompt=prompt, key=key, response_format=cited_statements_format)
    prompt2 = get_citations_from_stmts_prompt(summary=relevant_summary2, statements=statements2)
    key2 = f"TEST_structured_new__cited_stmt_gen___claude-sonnet-4__{query2}"
    append_to_gemini_batch_file(batch_filepath=batch_filepath, prompt=prompt2, key=key2, response_format=cited_statements_format)
    batch_job = make_gemini_batch_request(batch_filepath=batch_filepath)

    with open(os.path.join("batch_cited_stmt_gen_test", f"batchjobname__structured_new_{num}.txt"), 'w', encoding='utf-8') as f:
        f.write(batch_job.name)



def test_structured_new_batch_file_request_2():
    num = "2"

    query = "How does the success of providing artificial nest sites for solitary bees compare to providing them for bumblebees?"
    relevant_summary = "The success of providing artificial nest sites differs significantly between solitary bees and bumblebees, with solitary bees showing much higher success rates. For solitary bees, 29 out of 30 replicated trials showed occupancy by bees across 10 countries including Europe, North and South America and Asia (Action 47). Occupancy rates for solitary bee nest boxes, where reported, were between 1 and 26% of available cavities (Action 80). The number of occupied solitary bee nests can double over three years with repeated nest box provision at a given site, as demonstrated in three trials on agricultural land (Action 47). One study estimated that the number of foraging Osmia bees had increased in fields with nest boxes compared to fields without nest boxes (Action 47). In contrast, bumblebee nest box success is much lower. Three replicated trials since 1989 in the UK showed very low uptake rates of 0-2.5% for various nest box designs, while seven earlier trials in the UK, USA or Canada showed overall uptake rates between 10% and 48% (Action 48). Two replicated trials tested bumblebee nest boxes and both found very low uptake of 2% or less (Action 80). Underground nest boxes buried 5-10 cm underground with entrance pipes are generally the most effective for bumblebees, with seven replicated trials finding between 6% and 58% occupancy, but even these rates are often lower than solitary bee success rates (Action 48). However, there is no evidence captured for the effects of providing nest boxes on bumblebee populations (Action 48).",
    statements = [
        "The success of providing artificial nest sites differs significantly between solitary bees and bumblebees.",
        "Solitary bees show much higher success rates in occupying artificial nest sites than bumblebees do.",
        "For solitary bees, 29 out of 30 replicated trials showed occupancy by bees across 10 countries, including Europe, North and South America, and Asia.",
        "Reported occupancy rates for solitary bee nest boxes were between 1% and 26% of available cavities.",
        "The number of occupied solitary bee nests can double over three years with repeated nest box provision at a given site.",
        "The doubling of occupied solitary bee nests was demonstrated in three trials on agricultural land.",
        "One study estimated that the number of foraging Osmia bees had increased in fields with nest boxes compared to fields without nest boxes.",
        "In contrast to solitary bees, bumblebee nest box success is much lower.",
        "Three replicated trials since 1989 in the UK showed very low uptake rates of 0-2.5% for various bumblebee nest box designs.",
        "Seven earlier trials in the UK, USA, or Canada showed overall bumblebee nest box uptake rates between 10% and 48%.",
        "Two replicated trials that tested bumblebee nest boxes both found very low uptake of 2% or less.",
        "Underground nest boxes buried 5-10 cm underground with entrance pipes are generally the most effective for bumblebees.",
        "Seven replicated trials of underground bumblebee nest boxes found between 6% and 58% occupancy.",
        "The occupancy rates for even the most effective bumblebee nest boxes are often lower than solitary bee success rates.",
        "There is no evidence captured for the effects of providing nest boxes on bumblebee populations."
    ]

    query2 = "What actions related to reducing farming intensity can be taken to benefit wild bees?"
    relevant_summary2 = "Several actions related to reducing farming intensity can benefit wild bees. Four replicated trials in Europe have compared farmland meadows managed extensively with conventionally farmed meadows, with two trials finding enhanced numbers and diversity of wild bees on meadows with a delayed first cut and little agrochemical use (Action 22). Reducing grazing intensity on pastures can increase the abundance of cavity-nesting bees and wasps, as shown by one replicated trial demonstrating that reducing summer cattle grazing intensity increased bee abundance (Action 23). \n\nEvidence on reduced tillage shows mixed results - two replicated trials on squash farms in the USA had contrasting outcomes, with one showing no difference in bee abundance between tilled and untilled farms, while another found three times more squash bees on no-till farms than on conventional farms (Action 11).\n\nConverting to organic farming shows equivocal evidence - three replicated trials in Europe or Canada demonstrated higher wild bee abundance under organic arable farming than conventional farming, with one trial showing higher bee diversity in organically farmed wheat fields. However, three other replicated trials in Europe or the USA found no significant difference in bumblebee numbers, bumblebee species, or wild bees visiting flowering crops between conventional and organic farms (Action 25).\n\nRegarding pesticide reduction, one replicated trial in the USA showed that numbers of foraging bees on squash farms are not affected by the responsible use of pesticides (Action 27). Agri-environment schemes have shown variable results - four replicated trials in Europe demonstrated enhanced diversity and/or abundance of foraging wild bees under various European agri-environment schemes compared to conventionally-managed fields, while four other replicated trials found no enhancement in bee numbers or species under agri-environment schemes (Action 24).",
    statements2 = [
        "Several actions related to reducing farming intensity can benefit wild bees.",
        "Two replicated trials in Europe found enhanced numbers and diversity of wild bees on farmland meadows with a delayed first cut and little agrochemical use compared to conventionally farmed meadows.",
        "Reducing grazing intensity on pastures can increase the abundance of cavity-nesting bees and wasps.",
        "One replicated trial demonstrated that reducing summer cattle grazing intensity increased bee abundance.",
        "Evidence on the effect of reduced tillage on bees shows mixed results.",
        "One replicated trial on squash farms in the USA showed no difference in bee abundance between tilled and untilled farms.",
        "Another replicated trial on squash farms in the USA found three times more squash bees on no-till farms than on conventional farms.",
        "Converting to organic farming shows equivocal evidence for benefitting wild bees.",
        "Three replicated trials in Europe or Canada demonstrated higher wild bee abundance under organic arable farming than under conventional farming.",
        "One trial showed higher bee diversity in organically farmed wheat fields.",
        "Three other replicated trials in Europe or the USA found no significant difference in bumblebee numbers, bumblebee species, or wild bees visiting flowering crops between conventional and organic farms.",
        "One replicated trial in the USA showed that numbers of foraging bees on squash farms are not affected by the responsible use of pesticides.",
        "Agri-environment schemes have shown variable results for wild bees.",
        "Four replicated trials in Europe demonstrated enhanced diversity and/or abundance of foraging wild bees under various European agri-environment schemes compared to conventionally-managed fields.",
        "Four other replicated trials found no enhancement in bee numbers or species under agri-environment schemes."
    ]

    batch_filepath = os.path.join("batch_cited_stmt_gen_test", f"structured_new_request_file{num}.jsonl")
    cited_statements_format = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "statement": {"type": "STRING"},
                "citations": {
                    "type": "ARRAY",
                    "items": {"type": "STRING"}
                },
            },
            "required": ["statement", "citations"]
        }
    }

    prompt = get_citations_from_stmts_prompt(summary=relevant_summary, statements=statements)
    key = f"TEST_structured_new__cited_stmt_gen___claude-sonnet-4__{query}"
    append_to_gemini_batch_file(batch_filepath=batch_filepath, prompt=prompt, key=key, response_format=cited_statements_format)
    prompt2 = get_citations_from_stmts_prompt(summary=relevant_summary2, statements=statements2)
    key2 = f"TEST_structured_new__cited_stmt_gen___claude-sonnet-4__{query2}"
    append_to_gemini_batch_file(batch_filepath=batch_filepath, prompt=prompt2, key=key2, response_format=cited_statements_format)
    batch_job = make_gemini_batch_request(batch_filepath=batch_filepath)

    with open(os.path.join("batch_cited_stmt_gen_test", f"batchjobname__structured_new_{num}.txt"), 'w', encoding='utf-8') as f:
        f.write(batch_job.name)



def test_structured_new_batch_file_request_3():
    num = "3"

    query = "How does the success of providing artificial nest sites for solitary bees compare to providing them for bumblebees?"
    relevant_summary = "The success of providing artificial nest sites differs significantly between solitary bees and bumblebees, with solitary bees showing much higher success rates. For solitary bees, 29 out of 30 replicated trials showed occupancy by bees across 10 countries including Europe, North and South America and Asia (Action 47). Occupancy rates for solitary bee nest boxes, where reported, were between 1 and 26% of available cavities (Action 80). The number of occupied solitary bee nests can double over three years with repeated nest box provision at a given site, as demonstrated in three trials on agricultural land (Action 47). One study estimated that the number of foraging Osmia bees had increased in fields with nest boxes compared to fields without nest boxes (Action 47). In contrast, bumblebee nest box success is much lower. Three replicated trials since 1989 in the UK showed very low uptake rates of 0-2.5% for various nest box designs, while seven earlier trials in the UK, USA or Canada showed overall uptake rates between 10% and 48% (Action 48). Two replicated trials tested bumblebee nest boxes and both found very low uptake of 2% or less (Action 80). Underground nest boxes buried 5-10 cm underground with entrance pipes are generally the most effective for bumblebees, with seven replicated trials finding between 6% and 58% occupancy, but even these rates are often lower than solitary bee success rates (Action 48). However, there is no evidence captured for the effects of providing nest boxes on bumblebee populations (Action 48).",
    statements = [
        "The success of providing artificial nest sites differs significantly between solitary bees and bumblebees.",
        "Solitary bees show much higher success rates in occupying artificial nest sites than bumblebees do.",
        "For solitary bees, 29 out of 30 replicated trials showed occupancy by bees across 10 countries, including Europe, North and South America, and Asia.",
        "Reported occupancy rates for solitary bee nest boxes were between 1% and 26% of available cavities.",
        "The number of occupied solitary bee nests can double over three years with repeated nest box provision at a given site.",
        "The doubling of occupied solitary bee nests was demonstrated in three trials on agricultural land.",
        "One study estimated that the number of foraging Osmia bees had increased in fields with nest boxes compared to fields without nest boxes.",
        "In contrast to solitary bees, bumblebee nest box success is much lower.",
        "Three replicated trials since 1989 in the UK showed very low uptake rates of 0-2.5% for various bumblebee nest box designs.",
        "Seven earlier trials in the UK, USA, or Canada showed overall bumblebee nest box uptake rates between 10% and 48%.",
        "Two replicated trials that tested bumblebee nest boxes both found very low uptake of 2% or less.",
        "Underground nest boxes buried 5-10 cm underground with entrance pipes are generally the most effective for bumblebees.",
        "Seven replicated trials of underground bumblebee nest boxes found between 6% and 58% occupancy.",
        "The occupancy rates for even the most effective bumblebee nest boxes are often lower than solitary bee success rates.",
        "There is no evidence captured for the effects of providing nest boxes on bumblebee populations."
    ]

    query2 = "What actions related to reducing farming intensity can be taken to benefit wild bees?"
    relevant_summary2 = "Several actions related to reducing farming intensity can benefit wild bees. Four replicated trials in Europe have compared farmland meadows managed extensively with conventionally farmed meadows, with two trials finding enhanced numbers and diversity of wild bees on meadows with a delayed first cut and little agrochemical use (Action 22). Reducing grazing intensity on pastures can increase the abundance of cavity-nesting bees and wasps, as shown by one replicated trial demonstrating that reducing summer cattle grazing intensity increased bee abundance (Action 23). \n\nEvidence on reduced tillage shows mixed results - two replicated trials on squash farms in the USA had contrasting outcomes, with one showing no difference in bee abundance between tilled and untilled farms, while another found three times more squash bees on no-till farms than on conventional farms (Action 11).\n\nConverting to organic farming shows equivocal evidence - three replicated trials in Europe or Canada demonstrated higher wild bee abundance under organic arable farming than conventional farming, with one trial showing higher bee diversity in organically farmed wheat fields. However, three other replicated trials in Europe or the USA found no significant difference in bumblebee numbers, bumblebee species, or wild bees visiting flowering crops between conventional and organic farms (Action 25).\n\nRegarding pesticide reduction, one replicated trial in the USA showed that numbers of foraging bees on squash farms are not affected by the responsible use of pesticides (Action 27). Agri-environment schemes have shown variable results - four replicated trials in Europe demonstrated enhanced diversity and/or abundance of foraging wild bees under various European agri-environment schemes compared to conventionally-managed fields, while four other replicated trials found no enhancement in bee numbers or species under agri-environment schemes (Action 24).",
    statements2 = [
        "Several actions related to reducing farming intensity can benefit wild bees.",
        "Two replicated trials in Europe found enhanced numbers and diversity of wild bees on farmland meadows with a delayed first cut and little agrochemical use compared to conventionally farmed meadows.",
        "Reducing grazing intensity on pastures can increase the abundance of cavity-nesting bees and wasps.",
        "One replicated trial demonstrated that reducing summer cattle grazing intensity increased bee abundance.",
        "Evidence on the effect of reduced tillage on bees shows mixed results.",
        "One replicated trial on squash farms in the USA showed no difference in bee abundance between tilled and untilled farms.",
        "Another replicated trial on squash farms in the USA found three times more squash bees on no-till farms than on conventional farms.",
        "Converting to organic farming shows equivocal evidence for benefitting wild bees.",
        "Three replicated trials in Europe or Canada demonstrated higher wild bee abundance under organic arable farming than under conventional farming.",
        "One trial showed higher bee diversity in organically farmed wheat fields.",
        "Three other replicated trials in Europe or the USA found no significant difference in bumblebee numbers, bumblebee species, or wild bees visiting flowering crops between conventional and organic farms.",
        "One replicated trial in the USA showed that numbers of foraging bees on squash farms are not affected by the responsible use of pesticides.",
        "Agri-environment schemes have shown variable results for wild bees.",
        "Four replicated trials in Europe demonstrated enhanced diversity and/or abundance of foraging wild bees under various European agri-environment schemes compared to conventionally-managed fields.",
        "Four other replicated trials found no enhancement in bee numbers or species under agri-environment schemes."
    ]

    batch_filepath = os.path.join("batch_cited_stmt_gen_test", f"structured_new_request_file{num}.jsonl")
    cited_statements_format = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "statement": {"type": "STRING", "description": "One of the provided statements."},
                "citations": {
                    "type": "ARRAY",
                    "description": "List of citations (action numbers) that support the statement.",
                    "items": {"type": "STRING", "description": "An action number (e.g. 250) that supports the statement."}
                },
            },
            "required": ["statement", "citations"]
        }
    }

    prompt = get_citations_from_stmts_prompt(summary=relevant_summary, statements=statements)
    key = f"TEST_structured_new__cited_stmt_gen___claude-sonnet-4__{query}"
    append_to_gemini_batch_file(batch_filepath=batch_filepath, prompt=prompt, key=key, response_format=cited_statements_format)
    prompt2 = get_citations_from_stmts_prompt(summary=relevant_summary2, statements=statements2)
    key2 = f"TEST_structured_new__cited_stmt_gen___claude-sonnet-4__{query2}"
    append_to_gemini_batch_file(batch_filepath=batch_filepath, prompt=prompt2, key=key2, response_format=cited_statements_format)
    batch_job = make_gemini_batch_request(batch_filepath=batch_filepath)

    with open(os.path.join("batch_cited_stmt_gen_test", f"batchjobname__structured_new_{num}.txt"), 'w', encoding='utf-8') as f:
        f.write(batch_job.name)


def test_structured_new_batch_file_request_4():
    num = "4"

    query = "How does the success of providing artificial nest sites for solitary bees compare to providing them for bumblebees?"
    relevant_summary = "The success of providing artificial nest sites differs significantly between solitary bees and bumblebees, with solitary bees showing much higher success rates. For solitary bees, 29 out of 30 replicated trials showed occupancy by bees across 10 countries including Europe, North and South America and Asia (Action 47). Occupancy rates for solitary bee nest boxes, where reported, were between 1 and 26% of available cavities (Action 80). The number of occupied solitary bee nests can double over three years with repeated nest box provision at a given site, as demonstrated in three trials on agricultural land (Action 47). One study estimated that the number of foraging Osmia bees had increased in fields with nest boxes compared to fields without nest boxes (Action 47). In contrast, bumblebee nest box success is much lower. Three replicated trials since 1989 in the UK showed very low uptake rates of 0-2.5% for various nest box designs, while seven earlier trials in the UK, USA or Canada showed overall uptake rates between 10% and 48% (Action 48). Two replicated trials tested bumblebee nest boxes and both found very low uptake of 2% or less (Action 80). Underground nest boxes buried 5-10 cm underground with entrance pipes are generally the most effective for bumblebees, with seven replicated trials finding between 6% and 58% occupancy, but even these rates are often lower than solitary bee success rates (Action 48). However, there is no evidence captured for the effects of providing nest boxes on bumblebee populations (Action 48).",
    statements = [
        "The success of providing artificial nest sites differs significantly between solitary bees and bumblebees.",
        "Solitary bees show much higher success rates in occupying artificial nest sites than bumblebees do.",
        "For solitary bees, 29 out of 30 replicated trials showed occupancy by bees across 10 countries, including Europe, North and South America, and Asia.",
        "Reported occupancy rates for solitary bee nest boxes were between 1% and 26% of available cavities.",
        "The number of occupied solitary bee nests can double over three years with repeated nest box provision at a given site.",
        "The doubling of occupied solitary bee nests was demonstrated in three trials on agricultural land.",
        "One study estimated that the number of foraging Osmia bees had increased in fields with nest boxes compared to fields without nest boxes.",
        "In contrast to solitary bees, bumblebee nest box success is much lower.",
        "Three replicated trials since 1989 in the UK showed very low uptake rates of 0-2.5% for various bumblebee nest box designs.",
        "Seven earlier trials in the UK, USA, or Canada showed overall bumblebee nest box uptake rates between 10% and 48%.",
        "Two replicated trials that tested bumblebee nest boxes both found very low uptake of 2% or less.",
        "Underground nest boxes buried 5-10 cm underground with entrance pipes are generally the most effective for bumblebees.",
        "Seven replicated trials of underground bumblebee nest boxes found between 6% and 58% occupancy.",
        "The occupancy rates for even the most effective bumblebee nest boxes are often lower than solitary bee success rates.",
        "There is no evidence captured for the effects of providing nest boxes on bumblebee populations."
    ]

    query2 = "What actions related to reducing farming intensity can be taken to benefit wild bees?"
    relevant_summary2 = "Several actions related to reducing farming intensity can benefit wild bees. Four replicated trials in Europe have compared farmland meadows managed extensively with conventionally farmed meadows, with two trials finding enhanced numbers and diversity of wild bees on meadows with a delayed first cut and little agrochemical use (Action 22). Reducing grazing intensity on pastures can increase the abundance of cavity-nesting bees and wasps, as shown by one replicated trial demonstrating that reducing summer cattle grazing intensity increased bee abundance (Action 23). \n\nEvidence on reduced tillage shows mixed results - two replicated trials on squash farms in the USA had contrasting outcomes, with one showing no difference in bee abundance between tilled and untilled farms, while another found three times more squash bees on no-till farms than on conventional farms (Action 11).\n\nConverting to organic farming shows equivocal evidence - three replicated trials in Europe or Canada demonstrated higher wild bee abundance under organic arable farming than conventional farming, with one trial showing higher bee diversity in organically farmed wheat fields. However, three other replicated trials in Europe or the USA found no significant difference in bumblebee numbers, bumblebee species, or wild bees visiting flowering crops between conventional and organic farms (Action 25).\n\nRegarding pesticide reduction, one replicated trial in the USA showed that numbers of foraging bees on squash farms are not affected by the responsible use of pesticides (Action 27). Agri-environment schemes have shown variable results - four replicated trials in Europe demonstrated enhanced diversity and/or abundance of foraging wild bees under various European agri-environment schemes compared to conventionally-managed fields, while four other replicated trials found no enhancement in bee numbers or species under agri-environment schemes (Action 24).",
    statements2 = [
        "Several actions related to reducing farming intensity can benefit wild bees.",
        "Two replicated trials in Europe found enhanced numbers and diversity of wild bees on farmland meadows with a delayed first cut and little agrochemical use compared to conventionally farmed meadows.",
        "Reducing grazing intensity on pastures can increase the abundance of cavity-nesting bees and wasps.",
        "One replicated trial demonstrated that reducing summer cattle grazing intensity increased bee abundance.",
        "Evidence on the effect of reduced tillage on bees shows mixed results.",
        "One replicated trial on squash farms in the USA showed no difference in bee abundance between tilled and untilled farms.",
        "Another replicated trial on squash farms in the USA found three times more squash bees on no-till farms than on conventional farms.",
        "Converting to organic farming shows equivocal evidence for benefitting wild bees.",
        "Three replicated trials in Europe or Canada demonstrated higher wild bee abundance under organic arable farming than under conventional farming.",
        "One trial showed higher bee diversity in organically farmed wheat fields.",
        "Three other replicated trials in Europe or the USA found no significant difference in bumblebee numbers, bumblebee species, or wild bees visiting flowering crops between conventional and organic farms.",
        "One replicated trial in the USA showed that numbers of foraging bees on squash farms are not affected by the responsible use of pesticides.",
        "Agri-environment schemes have shown variable results for wild bees.",
        "Four replicated trials in Europe demonstrated enhanced diversity and/or abundance of foraging wild bees under various European agri-environment schemes compared to conventionally-managed fields.",
        "Four other replicated trials found no enhancement in bee numbers or species under agri-environment schemes."
    ]

    batch_filepath = os.path.join("batch_cited_stmt_gen_test", f"structured_new_request_file{num}.jsonl")
    cited_statements_format = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "statement": {"type": "STRING", "description": "One of the provided statements."},
                "citations": {
                    "type": "ARRAY",
                    "description": "List of citations (action numbers) given for the statement. Can be empty if no citations are given in the summary for this statement.",
                    "items": {"type": "STRING", "description": "An action number (e.g. 250) that supports the statement."}
                },
            },
            "required": ["statement", "citations"]
        }
    }

    prompt = get_citations_from_stmts_prompt(summary=relevant_summary, statements=statements)
    key = f"TEST_structured_new__cited_stmt_gen___claude-sonnet-4__{query}"
    append_to_gemini_batch_file(batch_filepath=batch_filepath, prompt=prompt, key=key, response_format=cited_statements_format)
    prompt2 = get_citations_from_stmts_prompt(summary=relevant_summary2, statements=statements2)
    key2 = f"TEST_structured_new__cited_stmt_gen___claude-sonnet-4__{query2}"
    append_to_gemini_batch_file(batch_filepath=batch_filepath, prompt=prompt2, key=key2, response_format=cited_statements_format)
    batch_job = make_gemini_batch_request(batch_filepath=batch_filepath)

    with open(os.path.join("batch_cited_stmt_gen_test", f"batchjobname__structured_new_{num}.txt"), 'w', encoding='utf-8') as f:
        f.write(batch_job.name)




def test_structured_new_batch_file_response(num : str):
    client = get_genai_client()
    with open(os.path.join("batch_cited_stmt_gen_test", f"batchjobname__structured_new_{num}.txt"), 'r', encoding='utf-8') as f:
        job_name = f.read().strip()
    batch_job = client.batches.get(name=job_name)

    if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
        print(f"Batch job {batch_job.name} completed successfully.")

        result_file_name = batch_job.dest.file_name

        file_content = client.files.download(file=result_file_name)
        text_content = file_content.decode('utf-8')

        # Prepend new batch result to output file
        output_filepath = os.path.join("batch_cited_stmt_gen_test", f"structured_new_batch_results{num}.jsonl")
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(text_content)
    else:
        print("Batch job not completed:", batch_job.state.name)


def load_result_test_structured_new_batch_file_response(num: str):
    output_filepath = os.path.join("batch_cited_stmt_gen_test", f"structured_new_batch_results{num}.jsonl")
    cited_statements_list = []
    with open(output_filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            key = data["key"]
            response_raw = data["response"]["candidates"][0]["content"]["parts"][0]["text"]
            response_formatted = json.loads(response_raw)
            cited_statements_list.append({"key": key, "response": response_formatted})
    with open(os.path.join("batch_cited_stmt_gen_test", f"structured_new_batch_results_parsed{num}.json"), 'w', encoding='utf-8') as f:
        json.dump(cited_statements_list, f, indent=2)


def main():
    qu_types = ["answerable", "unanswerable"]
    filter_stages = ["passed", "failed"]
    retrieval_types = ["hybrid_cross-encoder"]
    answering_model_providers = [
        "_claude-sonnet-4",
        "_gemini-2-5-pro",
        "_gpt-5",
        "fireworks_kimi-k2-0905"
    ]
    run_full_process(
        qu_types=qu_types,
        filter_stages=filter_stages,
        retrieval_types=retrieval_types,
        summary_model_providers=answering_model_providers
    )


if __name__ == "__main__":
    # main()
    # examine_schema()
    # test_file_batch_request()

    # test_structured_new_batch_file_request_1() # got error no such field 'additionalProperties'
    
    # test_structured_new_batch_file_request_2() # didn't get an error, WORKED
    # test_structured_new_batch_file_response(num="2") # WORKED
    # load_result_test_structured_new_batch_file_response(num="2")

    # test_structured_new_batch_file_request_3() # worked with better format, but now all statements have citations even though in previous run they didn't. Could be due to example's influence against empty lists?
    # test_structured_new_batch_file_response(num="3")
    # load_result_test_structured_new_batch_file_response(num="3") 

    # test_structured_new_batch_file_request_4()
    # test_structured_new_batch_file_response(num="4")
    # load_result_test_structured_new_batch_file_response(num="4")

    # status = check_gemini_batch_status(batch_job_name="batches/hg2yjncay1ebdv07tp59r8it8eu7su899qnt")
    # print(status)

