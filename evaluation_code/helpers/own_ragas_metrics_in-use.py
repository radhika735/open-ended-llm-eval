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


class LlmAttemptsContext():
    def __init__(self, max_attempts):
        self.__max_attempts = max_attempts
        self.__current_attempts = 0
    
    def get_max_attempts(self):
        return self.__max_attempts

    def get_current_attempts(self):
        return self.__current_attempts
    
    def inc_current_attempts(self):
        self.__current_attempts += 1


### API CALLING.

def get_client():
    """
    Get an OpenAI client configured for Openrouter.

    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouter")
    base_url = "https://openrouter.ai/api/v1"
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )


def call_llm(messages, model, provider, max_attempts=1, attempts_context=None, max_tokens=15000, reasoning_effort=None, response_format=None):
    logging.info(f"Calling LLM with model: {model}, provider pinned to: {provider}.")
    if attempts_context is None:
        attempts_context = LlmAttemptsContext(max_attempts=max_attempts)
    print(f"making call, current attempts = {attempts_context.get_current_attempts()}")
    client = get_client()
    request_params = {
        "model":model,
        "messages": messages,
        "max_tokens": max_tokens,
        "extra_body": {
            "require_parameters": True,
            "reasoning":{
                "enabled": True,
                "effort": reasoning_effort if reasoning_effort is not None else "high"
            },
        }
    }

    if response_format is not None:
        request_params.update({"response_format": response_format})
    
    if provider is not None:
        request_params["extra_body"].update({
            "provider": {
                "order": [f"{provider}"], # Specify the single provider you want to pin
                "allow_fallbacks": False     # Set fallback to None to prevent routing elsewhere
            }
        })

    try:
        attempts_context.inc_current_attempts()
        print(f"here, change attempts num to {attempts_context.get_current_attempts()}")
        response = client.chat.completions.create(**request_params)
        print("response set")

        # Usage details for logging purposes:
        usage_details = response.usage
        print(f"Token usage: completion_tokens={usage_details.completion_tokens} (reasoning_tokens={usage_details.completion_tokens_details.reasoning_tokens}), prompt_tokens={usage_details.prompt_tokens}, total_tokens={usage_details.total_tokens}, cached_tokens={usage_details.prompt_tokens_details.cached_tokens}\n")

    except Exception as e:
        print(f"Exception during LLM call / usage display: {str(e)}")
        if attempts_context.get_current_attempts() < attempts_context.get_max_attempts():
            print(f"Current attempts {attempts_context.get_current_attempts()} is less than max attempts {attempts_context.get_max_attempts()} so retrying")
            return call_llm(
                messages=messages,
                model=model,
                provider=provider,
                attempts_context=attempts_context,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
                response_format=response_format
            )
        else:
            print(f"Run out of max attempts {attempts_context.get_max_attempts()}, rethrowing exception.")
            raise

    print("successful execution")
    return response



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
    return cited_statements



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

    
def test():
    context = ActionParsingContext(required_fields=["action_id", "action_title", "key_messages"])

    ## Loamy soils good answer
    # question = "What are the most effective ways to increase soil organic carbon on loamy soils?"
    # answer = "The most effective ways to increase soil organic carbon on loamy soils include growing cover crops, implementing reduced tillage practices, using crop rotation, applying organic amendments, and utilizing mixed organic-inorganic fertilizers.\n\nGrowing cover crops when fields are empty is particularly beneficial for increasing soil organic carbon on loamy soils (Action 898). Studies found increased soil carbon levels under cover crops, with further increases when legumes were included in the cover crop mix. Implementing reduced tillage or no-tillage practices significantly enhances soil organic carbon accumulation (Action 906). Twelve studies comparing no-tillage and conventionally tilled systems found consistently higher soil organic carbon in soils under reduced tillage systems, and the effectiveness is further enhanced when combined with cover cropping and manure application. Using crop rotation, especially when legumes are included, also proves beneficial (Action 857). Four studies found increased soil organic carbon under crop rotations, particularly when legumes were incorporated into the system. Applying mixed organic and inorganic amendments provides another effective approach (Action 902). Four controlled trials found more organic carbon in soils treated with mixed fertilizers compared to inorganic fertilizers alone. Additionally, applying manures and agricultural composts can increase soil carbon levels (Action 911), though this method requires careful consideration of potential trade-offs. Finally, formulated chemical compounds like nitrogen or phosphorus fertilizers can also contribute to soil organic matter increases (Action 909), with five of six studies showing increased soil organic matter when these compounds were applied to various soil types including loam."
    # action_ids_in_answer = ["898","906","857","902","911","909"]
    # oracle_ids = ["906","857","902","907","911"]

    ## Bears bad answer
    # question = "What are the most beneficial actions for reducing human-wildlife conflict with bears?"
    # answer = "The most effective bear conflict reduction strategies include deterrence techniques and preventing access to food sources, while translocation shows mixed results.\nScaring or deterring bears using projectiles, noisemakers, guard dogs, or unpleasant substances has proven beneficial in modifying bear behavior and reducing conflicts in human-occupied areas (ref: 2347). Preventing bears from accessing anthropogenic food sources like garbage, crops, and pet food through bear-proof containers or exclusion methods is also likely beneficial for conflict reduction (ref: 2346). Conditioned taste aversion, which involves adding illness-inducing agents to problem foods at non-residential sites like orchards or campsites, shows promise in creating food aversions (ref: 2384). Enforcement measures for bear-proof garbage disposal demonstrate unknown effectiveness due to limited evidence (ref: 2345). Translocation of habituated bears is less recommended because it often leads to trade-offs - bears may return to conflict sites or re-offend after relocation (ref: 2341)."
    # action_ids_in_answer = ['2341', '2345', '2346', '2347', '2384']
    # oracle_ids = ["2330","2336","2346","2347","2385"]
    # statements = [
    #     'The most effective bear conflict reduction strategies include deterrence techniques.', 
    #     'The most effective bear conflict reduction strategies include preventing access to food sources.', 
    #     'Translocation shows mixed results for reducing bear conflict.', 
    #     'Scaring or deterring bears using projectiles, noisemakers, guard dogs, or unpleasant substances has proven beneficial in modifying bear behavior.', 
    #     'Scaring or deterring bears has proven beneficial in reducing conflicts in human-occupied areas.', 
    #     'Preventing bears from accessing anthropogenic food sources like garbage, crops, and pet food is likely beneficial for conflict reduction.', 
    #     'Preventing access to food sources can be done through bear-proof containers or exclusion methods.', 
    #     'Conditioned taste aversion shows promise in creating food aversions in bears.', 
    #     'Conditioned taste aversion involves adding illness-inducing agents to problem foods at non-residential sites like orchards or campsites.', 
    #     'Enforcement measures for bear-proof garbage disposal have unknown effectiveness due to limited evidence.', 
    #     'Translocation of habituated bears is less recommended because it often leads to trade-offs.', 
    #     'Translocated bears may return to conflict sites.', 
    #     'Translocated bears may re-offend after relocation.'
    # ]
    # statements = [
    #     "The most effective bear conflict reduction strategies include deterrence techniques and preventing access to food sources, while translocation shows mixed results.",
    #     "Scaring or deterring bears using projectiles, noisemakers, guard dogs, or unpleasant substances has proven beneficial in modifying bear behavior and reducing conflicts in human-occupied areas.",
    #     "Preventing bears from accessing anthropogenic food sources like garbage, crops, and pet food through bear-proof containers or exclusion methods is also likely beneficial for conflict reduction.",
    #     "Conditioned taste aversion, which involves adding illness-inducing agents to problem foods at non-residential sites like orchards or campsites, shows promise in creating food aversions.",
    #     "Enforcement measures for bear-proof garbage disposal demonstrate unknown effectiveness due to limited evidence.",
    #     "Translocation of habituated bears is less recommended because it often leads to trade-offs - bears may return to conflict sites or re-offend after relocation."
    # ]

    ## Bears good summary
    question = "What are the most beneficial actions for reducing human-wildlife conflict with bears?"
    answer = "Evidence indicates several actions can reduce bear-related conflicts. Diversionary feeding reduced nuisance behaviour by black bears in two before-and-after studies, and brown bears in Slovenia obtained 22\u201363% of annual dietary energy from provided food (2323). Scaring/deterrence had mixed outcomes: some studies found noise/pain deterrents did not prevent black bears returning to urban or human-occupied sites, while other studies found such actions deterred bears from seeking food; chasing nuisance black bears with dogs caused them to stay away longer; an electric fence prevented polar bear entry to a compound; chemical and acoustic repellents did not deter polar bears from baits in most cases (2347). Preventing access to food sources with electric shock devices stopped American black bears from accessing or damaging bird feeders (2346). Conditioned taste aversion led black bears to avoid treated foods (2384). Issuing enforcement notices requiring appropriate dumpster use did not reduce garbage accessibility to black bears (2345). Translocating problem or habituated bears often resulted in bears returning to capture locations and/or continuing nuisance, and for grizzly and black bears reduced survival compared to non-translocated bears; however, one controlled study found translocated brown bears occurred less frequently inside high potential conflict areas than non-translocated bears (2336, 2341)."
    action_ids_in_answer = ["2323","2336","2341","2345","2346","2347","2384"]
    oracle_ids = ["2330","2336","2346","2347","2385"]
    # statements = [
    #     'Evidence indicates several actions can reduce bear-related conflicts.',
    #     'Diversionary feeding reduced nuisance behaviour by black bears in two before-and-after studies.',
    #     'Brown bears in Slovenia obtained 22–63%% of their annual dietary energy from provided food.',
    #     'Scaring and deterrence methods had mixed outcomes.', 
    #     'Some studies found that noise and pain deterrents did not prevent black bears from returning to urban or human-occupied sites.', 
    #     'Other studies found that scaring and deterrence actions deterred bears from seeking food.',
    #     'Chasing nuisance black bears with dogs caused them to stay away longer.',
    #     'An electric fence prevented a polar bear from entering a compound.', 
    #     'In most cases, chemical and acoustic repellents did not deter polar bears from baits.', 
    #     'Preventing access to food sources with electric shock devices stopped American black bears from accessing or damaging bird feeders.',
    #     'Conditioned taste aversion led black bears to avoid treated foods.', 
    #     'Issuing enforcement notices that required appropriate dumpster use did not reduce garbage accessibility for black bears.',
    #     'Translocating problem or habituated bears often resulted in the bears returning to their capture locations and/or continuing their nuisance behavior.',
    #     'Translocation reduced the survival of grizzly and black bears compared to non-translocated bears.', 
    #     'One controlled study found that translocated brown bears occurred less frequently inside high-potential conflict areas than non-translocated bears.'
    # ]

    ## Intentionally bad (partially irrelevant) answer generated to soils query
    # answer = "Evidence suggests several management practices influence soil properties on loamy soils. Adopting no-tillage or reduced tillage practices consistently resulted in the stratification and accumulation of organic carbon near the soil surface compared to conventional tillage, although effects on total carbon in the deeper soil profile were mixed (1114). The application of lime to acidic loamy soils successfully raised pH into the optimal range for common row crops, which improved nutrient availability but did not alter the soil's water holding capacity (1131). The inclusion of cover crops in rotations was found to increase soil organic carbon stocks in the topsoil in multiple meta-analyses, though the magnitude of effect varied by climate and species (1102). Precision irrigation technologies, such as drip systems, significantly reduced water consumption by 30-50% compared to furrow irrigation in arid and semi-arid regions, improving water use efficiency (1120). Direct application of organic amendments like compost or animal manure was a highly effective strategy, with one long-term study showing a doubling of soil carbon content over 20 years in treated plots (1125). Subsurface tile drainage systems were shown to improve soil aeration and trafficability in poorly drained loams, but had little effect on phosphorus leaching (1105)."


    # all_ids = list(set(action_ids_in_answer) | set(oracle_ids))

    # docs_str = get_oracle_actions_as_str(id_list=all_ids, context=context)
    # docs = get_oracle_actions(id_list=all_ids, context=context)
    # print(faithfulness(question=question, summary=answer, docs=docs_str))
    model = "google/gemini-2.5-pro"
    provider = None

    print("getting statements")
    statements = _get_statements(question=question, answer=answer, model=model, provider=provider)
    for metric in ["faithfulness", "citation_correctness"]:
        print("evaluating",metric)
        result = evaluate_metric(
            judge_model=model,
            judge_provider=provider,
            metric_name=metric, 
            question=question, 
            summary=answer, 
            summary_statements=statements, 
            action_ids_in_summary=action_ids_in_answer,
            oracle_ids=oracle_ids,
            context=context    
        )

    print(f"'score':{result['score']}")
    # print(f"\n'question': {question}")
    print(f"\n'statements':")
    for s in result["statements"]:
        print(f'"{s}"')
    # print(f"\n'citations':")
    # for c in result["citations"]:
    #     print(f'{c}')
    print(f"\n'verdicts': {result["verdicts"]}")
    print(f"\n'reasonings':")
    for r in result["reasonings"]:
        print(f'"{r}"')

    # statements = get_statements(question=question, answer=answer)
    # for s in statements:
    #     print(f'"{s}"')

    # result = answer_relevance(query=question, answer=answer, n=10)
    # print(f"'score':{result['score']}")
    # for q in result["questions"]:
    #     print(f'"{q}"')


def test_statements():
    start = time.monotonic()
    question =  "What evidence exists for the effectiveness of different strategies to combat white-nose syndrome in bats?"
    summary = "Two randomized, controlled studies evaluated treating little brown bats with a probiotic bacterium: in Canada, treatment at the time of white-nose syndrome infection increased survival and reduced disease symptoms in caged bats, whereas treatment 21 days prior did not increase survival and was associated with worse symptoms; in the USA, treatment increased survival for free-flying bats within a mine but did not increase survival for caged bats and resulted in similar disease severity to untreated bats (2008). No studies were found that evaluated vaccinating bats against the white-nose syndrome pathogen (1011), decontaminating clothing and equipment after entering caves (2006), restricting human access to bat caves (1010), or breeding bats in captivity to supplement wild populations affected by white-nose syndrome (2009).",
    model = "google/gemini-2.5-flash"
    provider = None
    statements = _get_statements(question=question, answer=summary, model=model, provider=provider)
    for s in statements:
        print(f'"{s}"')
    cited_statements = _get_citations_from_statements(summary=summary, statements=statements, model=model, provider=provider)
    for cs in cited_statements:
        print(f'"{cs}"')
    print(f"Time taken: {time.monotonic() - start} seconds")


def main():
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
    test_statements()




