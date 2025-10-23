import json
import os
import logging
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
import time

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



### CITED STATEMENT EXTRACTION


class CitedStatement(BaseModel):
    statement : str
    citations : list[str]


def get_citations_from_statements(summary, statements, model, provider):
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
        return None
    return cited_statements



### UTILITIES


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
        raise RetrievalError(f"Error decoding JSON from file {filepath}: {str(e)}.") from e
    except FileNotFoundError as e:
        raise RetrievalError(f"File {filepath} not found.") from e


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



### FULL GENERATION PIPELINE


def run_cited_stmt_gen_for_stmts_file(stmts_filepath, max_stmts, cited_stmts_filepath, judge_model, judge_provider):
    try:
        stmt_dicts = read_json_file(stmts_filepath)
    except RetrievalError as e:
        logging.error(f"Unable to load statements for cited statement generation from file {stmts_filepath}: {e}")
        return
    try:
        cited_stmt_dicts = read_json_file(cited_stmts_filepath)
    except RetrievalError as e:
        logging.error(f"Unable to load existing cited statements from file {cited_stmts_filepath}: {e}. Starting fresh cited statement generation for this file.")
        cited_stmt_dicts = []

    try:
        stmt_count = 0
        current_stmt_idx = -1

        while stmt_count < max_stmts:
            current_stmt_idx += 1
            if current_stmt_idx >= len(stmt_dicts):
                logging.info(f"Reached end of statements in file {stmts_filepath}. Stopping cited statement generation for this file.")
                break

            stmt_dict = stmt_dicts[current_stmt_idx]
            if stmt_dict["summary_details"]["summary_statements"] is None:
                logging.warning(f"Skipping statements to query {stmt_dict["question_details"]["query"]} in file {stmts_filepath} as it has None statements.")
                continue

            if stmt_dict["summary_details"].get("gen_cited_stmts_received", False):
                logging.debug(f"Skipping cited statement generation for statements to query {stmt_dict["question_details"]["query"]} in file {stmts_filepath} as it has already been done.")
                continue
            else:
                logging.info(f"Generating cited statements for statements to query {stmt_dict["question_details"]["query"]} in file {stmts_filepath}.")


            # find cited statement dict corresponding to this stmt dict, if it already exists

            cited_stmt_dict_found = False
            current_cited_stmt_idx = None
            for i, cited_stmt_dict in enumerate(cited_stmt_dicts):
                if (cited_stmt_dict["question_details"]["query"] == stmt_dict["question_details"]["query"] and
                    cited_stmt_dict["summary_details"]["relevant_summary"] == stmt_dict["summary_details"]["relevant_summary"]):
                    cited_stmt_dict_found = True
                    current_cited_stmt_idx = i
                    break
            if not cited_stmt_dict_found:
                query = stmt_dict["question_details"]["query"]
                all_relevant_qu_ids = stmt_dict["question_details"]["all_relevant_qu_ids"]
                regenerated_qu_ids = stmt_dict["question_details"]["regenerated_qu_ids"]
                summary_model = stmt_dict["summary_details"]["summary_model"]
                summary_provider = stmt_dict["summary_details"]["summary_provider"]
                relevant_summary = stmt_dict["summary_details"]["relevant_summary"]
                summary_action_ids = stmt_dict["summary_details"]["summary_action_ids"]
                summary_stmts_model = stmt_dict["summary_details"]["summary_statements_model"]
                summary_stmts = stmt_dict["summary_details"]["summary_statements"]
                
            else:
                cited_stmt_dict = cited_stmt_dicts[current_cited_stmt_idx]
                query = cited_stmt_dict["question_details"]["query"]
                all_relevant_qu_ids = cited_stmt_dict["question_details"]["all_relevant_qu_ids"]
                regenerated_qu_ids = cited_stmt_dict["question_details"]["regenerated_qu_ids"]
                summary_model = cited_stmt_dict["summary_details"]["summary_model"]
                summary_provider = cited_stmt_dict["summary_details"]["summary_provider"]
                relevant_summary = cited_stmt_dict["summary_details"]["relevant_summary"]
                summary_stmts = cited_stmt_dict["summary_details"]["summary_statements"]
                summary_action_ids = cited_stmt_dict["summary_details"]["summary_action_ids"]
                summary_stmts_model = cited_stmt_dict["summary_details"]["summary_statements_model"]
                summary_stmts = cited_stmt_dict["summary_details"]["summary_statements"]

            cited_stmts = get_citations_from_statements(
                summary=relevant_summary,
                statements=summary_stmts,
                model=judge_model,
                provider=judge_provider
            )

            if cited_stmts is not None:
                if cited_stmt_dict_found:
                    cited_stmt_dicts[current_cited_stmt_idx]["summary_details"]["cited_statements_model"] = judge_model
                    cited_stmt_dicts[current_cited_stmt_idx]["summary_details"]["cited_statements"] = cited_stmts
                else:
                    new_cited_stmt_dict = {
                        "question_details": {
                            "query": query,
                            "all_relevant_qu_ids": all_relevant_qu_ids,
                            "regenerated_qu_ids": regenerated_qu_ids
                        },
                        "summary_details": {
                            "summary_model": summary_model,
                            "summary_provider": summary_provider,
                            "relevant_summary": relevant_summary,
                            "summary_action_ids": summary_action_ids,
                            "summary_statements_model": summary_stmts_model,
                            "summary_statements": summary_stmts,
                            "cited_statements_model": judge_model,
                            "cited_statements": cited_stmts
                        }
                    }
                    cited_stmt_dicts.append(new_cited_stmt_dict)
                    
                stmt_dicts[current_stmt_idx]["summary_details"]["gen_cited_stmts_received"] = True
                stmt_count += 1

    finally:
        if stmt_count > 0:
            # write the new cited statements to cited statement output file
            write_to_json_file(data_list=cited_stmt_dicts, filepath=cited_stmts_filepath)
            logging.info(f"Wrote cited statements for {stmt_count} statements to cited statements file {cited_stmts_filepath}.")
            # overwrite the statements file (it will contain the updated gen_cited_stmts_received flag)
            write_to_json_file(data_list=stmt_dicts, filepath=stmts_filepath)
            logging.info(f"Updated statements file {stmts_filepath} gen_cited_stmts_received fields.")
        else:
            logging.info(f"No new cited statements generated for statements file {stmts_filepath}.")


def run_cited_stmt_gen_for_stmts_dir(stmts_dir, cited_stmts_out_dir, judge_model, judge_provider, judging_metrics, offset_to_first_stmt_file=0, max_stmt_files=1, max_stmts_per_file=1):
    if not os.path.exists(stmts_dir):
        logging.error(f"Statements directory {stmts_dir} does not exist.")
        return
    else:
        logging.info(f"Starting cited statement generation for statements in directory: {stmts_dir}")
        stmts_filenames = [name for name in sorted(os.listdir(stmts_dir)) if name.endswith(".json")]
        
        for stmts_filename in stmts_filenames[offset_to_first_stmt_file:offset_to_first_stmt_file+max_stmt_files]:
            cited_stmts_filename = stmts_filename.replace("statements", "cited_statements")
            run_cited_stmt_gen_for_stmts_file(
                stmts_filepath = os.path.join(stmts_dir, stmts_filename),
                cited_stmts_filepath = os.path.join(cited_stmts_out_dir, cited_stmts_filename),
                judge_model = judge_model,
                judge_provider = judge_provider,
                max_stmts = max_stmts_per_file,
            )



def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='logfiles/cited_statement_gen_realtime.log')
    QU_TYPES = ["answerable, unanswerable"]
    FILTER_STAGES = ["passed", "failed"]
    



if __name__ == "__main__":
    main()