# some parts copied heavily from "Evaluation of RAG Metrics for Question Answering in the Telecom Domain" paper
import os
import json
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

### OPENAI BATCH ENDPOINT CALLING:

def get_openai_client():
    """
    Get an OpenAI client configured for OpenAI.

    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI client")
    return OpenAI(
        api_key=api_key,
    )


def write_openai_batch_file(batch_filepath, custom_id, messages, judge_model="gpt-5", max_tokens=15000, response_format=None):
    request_body = {
        "model": judge_model,
        "messages": messages,
        "max_tokens": max_tokens
    }

    if response_format is not None:
        request_body.update({"response_format": response_format})

    request_obj = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": request_body
    }

    if not batch_filepath.endswith(".jsonl"):
        raise ValueError("OpenAI batch endpoint only supports .jsonl files currently.")
    with open(batch_filepath, 'a') as f:
        f.write(json.dumps(request_obj) + "\n")
    logging.debug(f"Wrote OpenAI batch request with custom id {custom_id} to file {batch_filepath}.")


def make_openai_batch_request(batch_filepath):
    if not batch_filepath.endswith(".jsonl"):
        raise ValueError("OpenAI batch endpoint must be given .jsonl file type.")
    client = get_openai_client()
    batch_input_file = client.files.create(
        file=open(batch_filepath, "rb"),
        purpose="batch"
    )
    print("\n\nOpenAI batch input file:", batch_input_file)
    batch_input_file_id = batch_input_file.id
    print("\n\nOpenAI batch input file ID:", batch_input_file_id)
    batch_obj = client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
            "description": "nightly eval job"
        }
    )
    print("\n\nBatch metadata:", batch_obj)

    return batch_obj


def check_openai_batch_status(batch_obj):
    client = get_openai_client()
    batch = client.batches.retrieve(batch_obj.id)
    return batch

    
def get_openai_batch_results(batch_obj):
    client = get_openai_client()
    file_response = client.files.content(batch_obj.output_file_id)
    print(file_response.text)
    return file_response


### GEMINI BATCH ENDPOINT CALLING:
def get_genai_client():
    api_key = os.getenv("PAID_GEMINI_API_KEY")
    if not api_key:
        raise ValueError("PAID_GEMINI_API_KEY environment variable is required for Gemini client for evaluation generation")
    return genai.Client(
        api_key=api_key
    )


def write_gemini_batch_file(batch_filepath, key, prompt, response_format=None, max_reasoning_tokens=8192):
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
        request_body["config"] = config.to_dict()
    elif hasattr(config, 'to_json'):
        request_body["config"] = json.loads(config.to_json())
    else:
        plain_config = {
            "thinking_config": {"thinking_budget": max_reasoning_tokens}
        }
        if response_format is not None:
            plain_config.update({
                "response_mime_type": "application/json",
                "response_schema": response_format
            })
        request_body["config"] = plain_config
    
    
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
    ])
    if batch_job.state.name not in completed_states:
        logging.info(f"Current state: {batch_job.state.name}")
    else:
        logging.info(f"Job finished with state: {batch_job.state.name}")
        if batch_job.state.name == 'JOB_STATE_FAILED':
            logging.error(f"Error: {batch_job.error}")
    return batch_job


def get_gemini_batch_results(batch_job_name, output_filepath):
    client = get_genai_client()
    batch_job = client.batches.get(name=batch_job_name)
    if batch_job.state.name == 'JOB_STATE_SUCCEEDED':
        # If batch job was created with a file
        if batch_job.dest and batch_job.dest.file_name:
            # Results are in a file
            result_file_name = batch_job.dest.file_name
            logging.info(f"Results are in file: {result_file_name}")

            logging.info("Downloading result file content...")
            file_content = client.files.download(file=result_file_name)
            
            text_content = file_content.decode('utf-8')
            with open(output_filepath, 'w', encoding='utf-8') as f:
                f.write(text_content)

    else:
        print(f"Batch job not completed. Current state: {batch_job.state.name}")


def check_num_open_gemini_batch_jobs():
    client = get_genai_client()
    open_states = set([
        'JOB_STATE_PENDING',
        'JOB_STATE_RUNNING',
        'JOB_STATE_QUEUED',
    ])
    batch_jobs = client.batches.list(config={"page_size": 10})
    for batch_job in batch_jobs:
        print(f"Batch job: {batch_job.name}, state {batch_job.state}")

# check openai rate limits how many credits need to put on there.



### STATEMENT EXTRACTION.

def _get_statements(question, answer, model, provider):
    print("Getting statements.")
    ## RAGAS statement splitting prompt:
    # prompt = f"""
    #     Given a question and answer, create one or more statements from each sentence in the given answer. Output the statements in the following format strictly. [Statement n]: ... where the n is the statement number.\nquestion: {question}\nanswer: {answer}\nStatements:\n
    # """.strip()

    ## Own prompt:
    prompt = f"""
Given a question and answer, create one or more statements from each sentence in the given answer. The answer will likely have references of numerical document ids especially at the end of sentences - YOU MUST NOT MENTION THESE REFERENCES OR NUMERICAL IDS IN ANY OF THE STATEMENTS YOU EXTRACT. Each statement should be fully understandable by itself, which means it needs to be self-contained. This means there should be no pronouns in the statements. Output the statements in the following format strictly. [Statement n]: ... where the n is the statement number.\nquestion: {question}\nanswer: {answer}\nStatements:\n
    """.strip()

    messages=[
        {"role": "user", "content": prompt}
    ]

    response = call_llm(messages=messages, model=model, provider=provider).choices[0].message.content

    # Splitting statements
    response = re.sub(
        r"^.*?[0-9]+\]*[:\.]\s(.*?)", r"\1", response, flags=re.MULTILINE
    ).strip()
    statements = sent_tokenize(response)
    return statements


def get_statements_prompt(question, answer):
    print("Getting statements.")
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



class CitedStatement(BaseModel):
    statement : str
    citations : list[str]


def _get_citations_from_statements(summary, statements, model, provider):
    print("Getting citations from statements.")
    prompt = f"""
Given a summary of information and a list of statements extracted from this summary, you must extract the document IDs cited in the summary for each statement. There may be zero or more cited IDs per statement.
Output the statements with their extracted citations as a list of JSON objects.

Summary: {summary}
Statements: {statements}
    """.strip()

    messages = [
        {"role": "user", "content": prompt}
    ]

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

    llm_response = call_llm(messages=messages, response_format=cited_statements_response_format, model=model, provider=provider)
    response_content = llm_response.choices[0].message.content
    try:
        cited_statements = json.loads(response_content)
    except json.JSONDecodeError as e:
        logging.error(f"During citation extraction from statements, error decoding JSON response from model {model} with provider {provider}: {str(e)}. Response was: {response_content}")
        raise e
    return 


def get_citations_from_stmts_prompt_and_format(summary, statements):
    prompt = f"""
Given a summary of information and a list of statements extracted from this summary, you must extract the document IDs cited in the summary for each statement. There may be zero or more cited IDs per statement.
Output the statements with their extracted citations as a list of JSON objects.

Summary: {summary}
Statements: {statements}
    """.strip()

    return prompt, list[CitedStatement]



### METRIC EVALUATION. Assessing metrics: faithfulness, answer relevance, citation correctness.


## JSON Schema wrapping individual judgement:
class StatementVerdict(BaseModel):
    statement: str = Field(description="A given statement.")
    reasoning: str = Field(description="The reasoning behind the verdict.")
    verdict: str = Field(description="The verdict for the statement (i.e., 'Yes' or 'No').")


## Existing RAGAS faithfulness metric:
def _faithfulness(docs, statements, judge_model, judge_provider):
    print("Evaluating faithfulness")
    statements_str = "\n".join([f"[Statement {i+1}: {s}]" for i, s in enumerate(statements)])

    ## True RAGAS prompt:
#     prompt = f"""
# Consider the given context and following statements, then determine whether they are supported by the information present in the context. Provide a brief explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order. Output the in-order list of statements, a list of their verdicts in-order, and a list of the reasonings for the verdicts in-order as a JSON object.\nContext: {docs}\nStatements: {statements_str}\nResponse:\n
#     """.strip()

    ## Own, RAGAS-inspired but modified prompt:
    prompt = f"""
Consider the given context and following statements, then determine whether they are supported by the information present in the context. ALL parts of the statement must be FULLY supported for it to be assigned a 'Yes' verdict. Provide a thorough and rigorous explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order. Output the statements, reasonings and verdicts as a valid list of JSON objects.\nContext: {docs}\nStatements: {statements_str}\nResponse:\n
    """.strip()

    messages = [
        {"role": "user", "content": prompt}
    ]

    ## Response format using JSON Schema wrapping individual judgements:
    verdicts_response_format = {
        "type" : "json_schema",
        "json_schema" : {
            "name" : "Statements, Verdicts and Reasonings",
            "strict" : True,
            "schema" : {
                "type": "array",
                "items" : StatementVerdict.model_json_schema()
            }
        }
    }
    raw_response = call_llm(messages=messages, response_format=verdicts_response_format, model=judge_model, provider=judge_provider)

    response = raw_response.choices[0].message.content
    try:
        response = json.loads(response)
    except json.JSONDecodeError as e:
        logging.error(f"During faithfulness eval, error decoding JSON response from judge model {judge_model} with provider {judge_provider}: {str(e)}. Response was: {response}")
        raise e

    ## Extracting output when using JSON Schema wrapping individual judgements:
    judgements = []
    for obj in response:
        judgements.append({
            "statement": obj["statement"],
            "reasoning": obj["reasoning"],
            "verdict": True if "yes" in obj["verdict"].lower() else False
        })
    verdicts = [obj["verdict"] for obj in judgements]

    num_supported_statements = sum(1 for v in verdicts if v)
    total_statements = len(verdicts)
    score = (num_supported_statements / total_statements) if total_statements > 0 else 0

    return {
        "score":score,
        "statement_judgements":judgements
    }


## JSON schema wrapping individual judgement:
class CitedStatementVerdict(BaseModel):
    statement: str = Field(description="A given statement.")
    citations: list[str] = Field(description="The list of citations for the statement (list).")
    reasoning: str = Field(description="The reasoning behind the verdict.")
    verdict: str = Field(description="The verdict for the statement (i.e., 'Yes' or 'No').")


## New metric: citation correctness within the summary.
#   Judges the "accuracy" of the citations themeselves for each statement, i.e. whether the citations given for each statement actually support the statement.
#   Statements with no citations are consequently judges as not supported by default, meaning unsupported statements are automatically penalised.
#   A high score is indicative of most facts being cited and those cited documents actually supporting the facts.
def _citation_correctness(summary, docs, statements, judge_model, judge_provider):
    print("Evaluating citation correctness")
    cited_statements = _get_citations_from_statements(summary=summary, statements=statements, model=judge_model, provider=judge_provider)
    statements_joined_citations = [{"statement":obj["statement"], "citations": ", ".join(obj["citations"])} for obj in cited_statements]
    cited_statements_str = "\n".join([f"[Statement {i+1}: {obj["statement"]}\nCitations for statement {i+1}: {obj["citations"]}]" for i, obj in enumerate(statements_joined_citations)])

    ## Least alterations to RAGAS faithfulness prompt:
    # prompt = f"""
    #     Consider the given context and following statements, then determine whether they are FULLY supported by the information present in the context CITED for the corresponding statement. If a statement has no citations then it is not supported by default. Provide a thorough explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order. Output the in-order list of statements, a list of the citations for each statement, a list of the statement verdicts in-order, and a list of the reasonings for the verdicts in-order as a JSON object.\nContext: {docs}\nStatements with their citations: {cited_statements_str}\nResponse:\n
    # """.strip()
#     prompt = f"""
# Consider the given context and following statements, then determine whether they are supported by the information present in the context CITED for the corresponding statement. ALL parts of the statement must be FULLY supported for it to be assigned a 'Yes' verdict. If a statement has no citations then it is not supported by default. Provide a thorough and rigorous explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order. Output the statements, the list of citations for each statement, reasonings and verdicts as a valid list of JSON objects.\nContext: {docs}\nStatements: {cited_statements_str}\nResponse:\n
#     """.strip()

    ## Based on own, RAGAS-inspired but modified, faithfulness prompt:
    prompt = f"""
Consider the given context and following statements, then determine whether they are supported by the information present in the context CITED for the corresponding statement. ALL parts of the statement must be FULLY supported for it to be assigned a 'Yes' verdict. If a statement has no citations then it is not supported by default. Provide a thorough and rigorous explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order. Output the statements, the list of citations for each statement, reasonings and verdicts as a valid list of JSON objects.\nContext: {docs}\nStatements: {cited_statements_str}\nResponse:\n
    """.strip()
    
    messages = [
        {"role": "user", "content": prompt}
    ]

    ## Response format using JSON schema wrapping individual judgements:
    verdicts_response_format = {
        "type" : "json_schema",
        "json_schema" : {
            "name" : "Statements, Citations, Verdicts and Reasonings",
            "strict" : True,
            "schema": {
                "type": "array",
                "items" : CitedStatementVerdict.model_json_schema()
            }
        }
    }

    raw_response = call_llm(messages=messages, response_format=verdicts_response_format, model=judge_model, provider=judge_provider)

    response = raw_response.choices[0].message.content
    try:
        response = json.loads(response)
    except json.JSONDecodeError as e:
        logging.error(f"During citation correctness eval, error decoding JSON response from judge model {judge_model} with provider {judge_provider}: {str(e)}. Response was: {response}")
        raise e

    judgements = []
    for obj in response:
        judgements.append({
            "statement": obj["statement"],
            "citations": obj["citations"],
            "reasoning": obj["reasoning"],
            "verdict": True if "yes" in obj["verdict"].lower() else False
        })
    verdicts = [obj["verdict"] for obj in judgements]
    
    num_supported_statements = sum(1 for v in verdicts if v)
    total_statements = len(verdicts)
    score = (num_supported_statements / total_statements) if total_statements > 0 else 0

    return {
        "score":score,
        "statement_judgements":judgements
    }


## New metric: relevance of the summary to the question.
#   Developed this metric due to not using answer_relevance metric.
#   This and a completeness metric would be a substitute to answer_relevance which takes into account both relevance of answer and completeness of it.
#   The metric works by splitting the summary into statements, and giving a verdict on each one as to whether it is relevant to answering the question (similar to RAGAS faithfulness procedure).
#   A high score is indicative of many statements in the summary being relevant to answering the question.
def _relevance(question, statements, judge_model, judge_provider):
    print("Evaluating relevance")
    statements_str = "\n".join([f"[Statement {i+1}: {s}]" for i, s in enumerate(statements)])
    prompt = f"""
Given a question and a list of statements, determine whether each statement is relevant to answering the question. Provide a thorough and rigorous explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order. Output the statements, the list of citations for each statement, reasonings and verdicts as a valid list of JSON objects.\nQuestion: {question}\nStatements: {statements_str}\nResponse:\n
    """.strip()

    messages = [
        {"role": "user", "content": prompt}
    ]

    verdicts_response_format = {
        "type" : "json_schema",
        "json_schema" : {
            "name" : "Statements, Verdicts and Reasonings",
            "strict" : True,
            "schema" : {
                "type": "array",
                "items" : StatementVerdict.model_json_schema()
            }
        }
    }
    raw_response = call_llm(messages=messages, response_format=verdicts_response_format, model=judge_model, provider=judge_provider)

    response = raw_response.choices[0].message.content
    try:
        response = json.loads(response)
    except json.JSONDecodeError as e:
        logging.error(f"During relevance eval, error decoding JSON response from judge model {judge_model} with provider {judge_provider}: {str(e)}. Response was: {response}")
        raise e

    judgements = []
    for obj in response:
        judgements.append({
            "statement": obj["statement"],
            "reasoning": obj["reasoning"],
            "verdict": True if "yes" in obj["verdict"].lower() else False
        })
    verdicts = [obj["verdict"] for obj in judgements]

    num_relevant_statements = sum(1 for v in verdicts if v)
    total_statements = len(verdicts)
    score = (num_relevant_statements / total_statements) if total_statements > 0 else 0

    return {
        "score":score,
        "statement_judgements":judgements
    }




### ACTION DOCS PARSING AND RETRIEVAL

def get_oracle_actions(id_list, context : ActionParsingContext):
    parsed_actions = []
    for id in id_list:
        parsed_actions.append(get_parsed_action_by_id(id=id, context=context))

    return parsed_actions



### FULL METRIC EVALUATION PIPELINE


def evaluate_metric(metric_name : str, question, summary, summary_statements, action_ids_in_summary, oracle_ids, context : ActionParsingContext, judge_model, judge_provider):
    # Getting the oracle actions
    oracle_actions = get_oracle_actions(id_list=oracle_ids, context=context)

    # Getting the actions cited in the summary
    cited_actions = get_oracle_actions(id_list=action_ids_in_summary, context=context)

    all_docs = oracle_actions + cited_actions
    all_docs_str = "\n\n".join([get_parsed_action_as_str(action=action) for action in all_docs])

    try:
        if metric_name == "faithfulness":
            result = _faithfulness(docs=all_docs_str, statements=summary_statements, judge_model=judge_model, judge_provider=judge_provider)
        elif metric_name == "citation_correctness":
            result = _citation_correctness(summary=summary, docs=all_docs_str, statements=summary_statements, judge_model=judge_model, judge_provider=judge_provider)
        elif metric_name == "relevance":
            result = _relevance(question=question, statements=summary_statements, judge_model=judge_model, judge_provider=judge_provider)
        else:
            raise ValueError(f"Unknown metric name: {metric_name}")
    except json.JSONDecodeError as e:
        logging.error(f"Skipping metric {metric_name} evaluation for question '{question}' due to JSON decoding error: {str(e)}.")
        return {
            "metric": metric_name,
            "evaluation": None
        }
    
    return {
        "metric": metric_name,
        "evaluation": result
    }


def assemble_evaluations(summary_obj, statements, evaluations, judge_model, judge_provider):
    question_details = {
        "query": summary_obj["query"],
        "all_relevant_qu_ids" : summary_obj["all_relevant_action_ids"],
        "regenerated_qu_ids": summary_obj["regenerated_ids"]
    }
    summary_details = {
        "summary_model": summary_obj["model"],
        "summary_provider": summary_obj["provider"],
        "relevant_summary": summary_obj["relevant_summary"],
        "summary_action_ids": summary_obj["summary_action_ids"]
    }
    evaluation_details = {
        "judge_model": judge_model,
        "judge_provider": judge_provider,
        "summary_statements": statements,
        "evaluations": evaluations
    }
    return {
        "question_details": question_details,
        "summary_details": summary_details,
        "evaluation_details": evaluation_details
    }


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


def write_to_json_file(data_list, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)
    except TypeError as e:
        logging.error(f"Error writing to JSON file {filepath}: {str(e)}.")


def get_metrics_not_done(eval_already_done, judge_model, judge_provider, judging_metrics):
    found = False
    for i, judge_model_metrics in enumerate(eval_already_done):
        prev_model = judge_model_metrics["judge_model"]
        prev_provider = judge_model_metrics["judge_provider"]
        if judge_model == prev_model and judge_provider == prev_provider:
            found = True
            # If the judge model/provider combo has already been done, check if all required metrics are done
            existing_metrics = judge_model_metrics["metrics"]
            missing_metrics = [m for m in judging_metrics if m not in existing_metrics]
            
    if not found:# If the judge model/provider has not been done before, all required metrics need to be done for this judge
        missing_metrics = judging_metrics.copy()

    return missing_metrics


def update_eval_by_models_metrics(eval_already_done, judge_model, judge_provider, new_metrics):
    found = False
    for i, judge_model_metrics in enumerate(eval_already_done):
        prev_model = judge_model_metrics["judge_model"]
        prev_provider = judge_model_metrics["judge_provider"]
        if judge_model == prev_model and judge_provider == prev_provider:
            found = True
            # If the judge model/provider combo has already been done, update the metrics list
            existing_metrics = judge_model_metrics["metrics"]
            existing_metrics.extend(new_metrics)
            break
    if not found:# If the judge model/provider has not been done before, add a new entry
        eval_already_done.append({
            "judge_model": judge_model,
            "judge_provider": judge_provider,
            "metrics": new_metrics
        })


def run_eval_for_summaries_file(summaries_filepath, max_summaries, eval_filepath, judge_model, judge_provider, judging_metrics, context : ActionParsingContext):
    try:
        file_summary_dicts = read_json_file(summaries_filepath)
    except RetrievalError as e:
        logging.error(f"Unable to load summaries for evaluation from file {summaries_filepath}: {e}")
        return
    try:
        file_eval_dicts = read_json_file(eval_filepath)
    except RetrievalError as e:
        logging.error(f"Unable to load existing evals from file {eval_filepath}: {e}. Starting fresh evaluation for this file.")
        file_eval_dicts = []

    
    summary_dicts = copy.deepcopy(file_summary_dicts)
    try:
        summary_count = 0
        current_summary_idx = -1

        while summary_count < max_summaries:
            current_summary_idx += 1
            if current_summary_idx >= len(summary_dicts):
                logging.info(f"Reached end of summaries in file {summaries_filepath}. Stopping evaluation for this file.")
                break
            summary_dict = summary_dicts[current_summary_idx]
            if summary_dict["relevant_summary"] is None:
                logging.warning(f"Skipping summary to query {summary_dict['query']} in file {summaries_filepath} as it has None summary.")
                continue

            eval_dict_found = False
            current_eval_idx = None
            for i, eval_dict in enumerate(file_eval_dicts):
                if (eval_dict["question_details"]["query"] == summary_dict["query"] and
                    eval_dict["summary_details"]["relevant_summary"] == summary_dict["relevant_summary"]):
                    eval_dict_found = True
                    current_eval_idx = i
                    break
            if not eval_dict_found:
                query = summary_dict["query"]
                all_relevant_qu_ids = summary_dict["all_relevant_action_ids"]
                summary_model = summary_dict["model"]
                summary_provider = summary_dict["provider"]
                relevant_summary = summary_dict["relevant_summary"]
                summary_action_ids = summary_dict["summary_action_ids"]
            else:
                eval_dict = file_eval_dicts[current_eval_idx]
                query = eval_dict["question_details"]["query"]
                all_relevant_qu_ids = eval_dict["question_details"]["all_relevant_qu_ids"]
                summary_model = eval_dict["summary_details"]["summary_model"]
                summary_provider = eval_dict["summary_details"]["summary_provider"]
                relevant_summary = eval_dict["summary_details"]["relevant_summary"]
                summary_action_ids = eval_dict["summary_details"]["summary_action_ids"]

            eval_by_models = summary_dict.get("eval_by_models", [])
            unevaluated_metrics = get_metrics_not_done(
                eval_already_done=eval_by_models, 
                judge_model=judge_model, 
                judge_provider=judge_provider, 
                judging_metrics=judging_metrics
            )

            if unevaluated_metrics:
                logging.info(f"Evaluating {unevaluated_metrics} for summary generated by model: {summary_model} and provider: {summary_provider} to query: {query}")
                if eval_dict_found:
                    summary_statements = eval_dict["evaluation_details"]["summary_statements"]
                else:
                    summary_statements = _get_statements(question=query, answer=relevant_summary, model=judge_model, provider=judge_provider)
                done_metrics = []
                evaluations = []
                for metric in unevaluated_metrics:
                    current_eval = evaluate_metric(
                            judge_model=judge_model,
                            judge_provider=judge_provider,
                            metric_name=metric,
                            question=query,
                            summary=relevant_summary,
                            summary_statements=summary_statements,
                            action_ids_in_summary=summary_action_ids,
                            oracle_ids=all_relevant_qu_ids,
                            context=context
                        )
                    if current_eval["evaluation"] is not None:
                        evaluations.append(current_eval)
                        done_metrics.append(metric)
                
                if evaluations != []:
                    if eval_dict_found:
                        file_eval_dicts[current_eval_idx]["evaluation_details"]["evaluations"].extend(evaluations)
                    else:
                        assembled_evaluation = assemble_evaluations(
                            summary_obj = summary_dict,
                            statements = summary_statements,
                            evaluations = evaluations,
                            judge_model = judge_model,
                            judge_provider = judge_provider
                        )
                        file_eval_dicts.append(assembled_evaluation)

                    update_eval_by_models_metrics(
                        eval_already_done=eval_by_models,
                        judge_model=judge_model,
                        judge_provider=judge_provider,
                        new_metrics=done_metrics
                    )
                    summary_dicts[current_summary_idx]["eval_by_models"] = eval_by_models
                    summary_count += 1
    
    finally:
        if summary_count > 0:
            # write the new evals to eval output file
            write_to_json_file(data_list=file_eval_dicts, filepath=eval_filepath)
            logging.info(f"Wrote evaluation for {summary_count} summaries to eval file {eval_filepath}.")
            # overwrite the summaries file (it will contain the updated eval_by_models field)
            write_to_json_file(data_list=summary_dicts, filepath=summaries_filepath)
            logging.info(f"Updated summaries file {summaries_filepath} eval_by_models fields.")
        else:
            logging.info(f"No new evaluations done for file {summaries_filepath}.")



def run_eval_for_summaries_dir(summaries_dir, eval_out_dir, judge_model, judge_provider, judging_metrics, context : ActionParsingContext, offset_to_first_summary_file=0, max_summary_files=1, max_summaries_per_file=1):
    if not os.path.exists(summaries_dir):
        logging.error(f"Summaries directory {summaries_dir} does not exist.")
        return
    else:
        logging.info(f"Starting evaluation of summaries in directory: {summaries_dir}")
        summaries_filenames = [name for name in sorted(os.listdir(summaries_dir)) if name.endswith(".json")]
        
        for summaries_filename in summaries_filenames[offset_to_first_summary_file:offset_to_first_summary_file+max_summary_files]:
            eval_filename = summaries_filename.replace("summaries", "eval")
            run_eval_for_summaries_file(
                summaries_filepath = os.path.join(summaries_dir, summaries_filename),
                eval_filepath = os.path.join(eval_out_dir, eval_filename),
                judge_model = judge_model,
                judge_provider = judge_provider,
                judging_metrics = judging_metrics,
                max_summaries = max_summaries_per_file,
                context = context
            )



def batch_gen_summary_statements_for_file(summaries_filepath, stmts_filepath, context : ActionParsingContext, max_summaries):
    try:
        file_summary_dicts = read_json_file(summaries_filepath)
    except RetrievalError as e:
        logging.error(f"Unable to load summaries for evaluation from file {summaries_filepath}: {e}")
        return
    try:
        file_stmt_dicts = read_json_file(stmts_filepath)
    except RetrievalError as e:
        logging.error(f"Unable to load existing statements from file {stmts_filepath}: {e}. Starting fresh statement generation for this file.")
        file_stmt_dicts = []
        
    summary_dicts = copy.deepcopy(file_summary_dicts)

    




def batch_gen_summary_statements_for_dir(summaries_dir, statements_out_dir, context : ActionParsingContext, offset_to_first_summary_file=0, max_summary_files=1, max_summaries_per_file=1):
    if not os.path.exists(summaries_dir):
        logging.error(f"Summaries directory {summaries_dir} does not exist.")
        return
    else:
        logging.info(f"Starting summary statements gen of summaries in directory : {summaries_dir}")
        summaries_filenames = [name for name in sorted(os.listdir(summaries_dir)) if name.endswith(".json")]

        for summaries_filename in summaries_filenames[offset_to_first_summary_file:offset_to_first_summary_file+max_summary_files]:
            statements_filename = summaries_filename.replace("summaries", "statements")
            batch_gen_summary_statements_for_file(
                summaries_filepath = os.path.join(summaries_dir, summaries_filename),
                stmts_filepath = os.path.join(statements_out_dir, statements_filename)
            )

# class LocalBatchObj():
#     def __init__(self, name, display_name, state, error, create_time, start_time, end_time, update_time, model, src, dest):
#         self.name = name
#         self.display_name = display_name
#         self.state = state
#         self.error = error
#         self.create_time = create_time
#         self.start_time = start_time
#         self.end_time = end_time
#         self.update_time = update_time
#         self.model = model
#         self.src = src
#         self.dest = dest

    
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


def gen_evals_realtime():
    logging.basicConfig(level=logging.INFO, filename="logfiles/own_ragas_metrics_in-use.log", format='%(asctime)s - %(levelname)s - %(message)s')
    # disable httpx logging
    logging.getLogger("httpx").setLevel(logging.WARNING)

    judge_model_provider_list = [
        ("google/gemini-2.5-pro", None),
        # ("google/gemini-2.5-flash", None),
        # ("openai/gpt-5", None),
        # ("openai/gpt-5-mini", None)
    ]

    CONTEXT = ActionParsingContext(required_fields=["action_id", "action_title", "key_messages"])

    QU_TYPES = ["answerable"]#, "unanswerable"]
    FILTER_STAGES = ["passed"]

    try:
        overall_start = time.monotonic()
        for qu_type in QU_TYPES:
            for filter_stage in FILTER_STAGES:
                for judge_model, judge_provider in judge_model_provider_list:
                    OFFSET_TO_FIRST_SUMMARY_FILE = 0
                    MAX_SUMMARY_FILES = 1
                    MAX_SUMMARIES_PER_FILE = 1

                    cleaned_judge_model = parse_model_name(judge_model)
                    cleaned_judge_provider = parse_provider_name(judge_provider)
                    cleaned_judge_name = f"{cleaned_judge_provider}_{cleaned_judge_model}"

                    answering_model_name_cleaned = "_gpt-5"

                    summaries_dir = f"live_summaries/{qu_type}_{filter_stage}_qus_summaries/hybrid_cross-encoder/eval_annotated/{answering_model_name_cleaned}"
                    eval_dir = f"live_evaluations/{qu_type}_{filter_stage}_evals/judge_{cleaned_judge_name}/summaries_{answering_model_name_cleaned}"

                    logging.info("STARTING evaluation process.")
                    internal_start = time.monotonic()
                    try:
                        run_eval_for_summaries_dir(
                            summaries_dir = summaries_dir,
                            eval_out_dir = eval_dir,
                            judge_model = judge_model,
                            judge_provider = judge_provider,
                            judging_metrics = ["faithfulness", "citation_correctness", "relevance"],
                            context = CONTEXT,
                            offset_to_first_summary_file=OFFSET_TO_FIRST_SUMMARY_FILE,
                            max_summary_files=MAX_SUMMARY_FILES,
                            max_summaries_per_file=MAX_SUMMARIES_PER_FILE
                        )
                    finally:
                        internal_time_taken = time.monotonic() - internal_start
                        print(f"Time taken for judge model: {judge_model} pinned to provider: {judge_provider} evaluating unanswerable {filter_stage} questions was: {internal_time_taken} seconds")

    except KeyboardInterrupt as e:
        logging.info(f"Keyboard interrupt: {e}")
    
    overall_time_taken = time.monotonic() - overall_start
    print(f"Time taken overall: {overall_time_taken} seconds")





if __name__ == "__main__":
    test_batch_request()




