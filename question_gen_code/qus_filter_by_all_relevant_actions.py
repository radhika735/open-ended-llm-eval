import json
from dotenv import load_dotenv
from google import genai 
from google.genai import types, errors
from google.api_core import exceptions
import logging
from pydantic import BaseModel
import os
import time

from utils.action_retrieval import get_synopsis_data_as_str
from utils.exceptions import RetrievalError, FatalAPIError, NonFatalAPIError, APIError, FileWriteError

load_dotenv()


class FilterContext():
    def __init__(self, qu_source_dir, max_calls, max_synopses):
        self.__qu_source_dir = qu_source_dir
        self.__max_calls = max_calls
        self.__current_calls = 0
        self.__max_synopses = max_synopses
        self.__current_synopses = 0

    def get_qu_source_dir(self):
        return self.__qu_source_dir

    def get_max_calls(self):
        return self.__max_calls

    def get_current_calls(self):
        return self.__current_calls

    def inc_current_calls(self):
        self.__current_calls += 1

    def get_max_synopses(self):
        return self.__max_synopses

    def get_current_synopses(self):
        return self.__current_synopses
    
    def inc_current_synopses(self):
        self.__current_synopses += 1


class RelevantActions(BaseModel):
    query: str
    relevant_action_ids: list[str]


def get_llm_relevant_actions(actions_data, query_list, synopsis, context : FilterContext, doc_type="bg_km"):
    prompt = f"""{actions_data}\n\n\n
    Above is a document containing conservation actions, their effectiveness and key messages relating to them. This has been gathered by hand and forms part of the Conservation Evidence living evidence database. Each action is prefixed with a numerical id.
    You are given a list of queries, each of which are answerable by using multiple actions in the document above. Your task is to return a list for each query, with each list containing the ids of ALL the actions that are relevant to that query. 
    You MUST find ALL the relevant actions for each query.
    Format your response as a list of JSON objects, with each object containing the query and the corresponding list of relevant action ids you have identified.

    Here are the queries: {query_list}
    """
    try:
        client = genai.Client()
        model_name = "gemini-2.5-pro"

        input_tokens = client.models.count_tokens(model=model_name, contents=prompt).total_tokens
        if input_tokens > 100000: # synopsis size (+ prompt) may exceed input token limit for request. Do not make api request.
            raise NonFatalAPIError(f"Total prompt for {synopsis} {doc_type} is {input_tokens} tokens long, exceeding input limit of 125,000 (empirically 100,000) tokens per minute.")
        
        if context.get_current_calls() >= context.get_max_calls():
            raise FatalAPIError("User-set MAX CALLS exceeded. Cannot make API call.")

        logging.info("Making API call.")
        context.inc_current_calls()
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=8192),
                response_mime_type="application/json",
                response_schema=list[RelevantActions]
            )
        )

        ids = [relevant_actions_obj.model_dump() for relevant_actions_obj in response.parsed]
        return ids

    except KeyError as e:
        raise NonFatalAPIError(f"Unexpected API response format: {str(e)}.")
    
    except TypeError as e:
        logging.error(f"Type error in API response: {str(e)}. Response content: {response.text if response else 'No response'}. Retrying request in 60 secs.")
        time.sleep(60)
        return get_llm_relevant_actions(actions_data=actions_data, query_list=query_list, synopsis=synopsis, doc_type=doc_type, context=context)


    except errors.ServerError as e:
        logging.error(f"Server side error, retrying request in 60 secs: {str(e)}")
        time.sleep(60)
        return get_llm_relevant_actions(actions_data=actions_data, query_list=query_list, synopsis=synopsis, doc_type=doc_type, context=context)

    
    except errors.ClientError as e:
        logging.error(f"Client side error: {str(e)}")
        
        if e.code == 429: # resource exhausted error (rate limit exceeded)
            error_dict = e.details
            error_id = error_dict["error"]["details"][0]["violations"][0]["quotaId"]

            if error_id == "GenerateRequestsPerDayPerProjectPerModel-FreeTier":
                raise FatalAPIError(f"Exceeded free tier quota of 50 requests per day.")

            elif error_id == "GenerateContentInputTokensPerModelPerMinute-FreeTier":
                if input_tokens >= 100000: # synopsis size itself may have exceeded input token size limit for free tier gemini (125000 input tokens / min), can't filter questions
                    # should not reach this branch - have added a check for this before making the API call.
                    raise NonFatalAPIError(f"Total prompt for {synopsis} {doc_type} is {input_tokens} tokens long, exceeding input limit of 125,000 (empirically 100,000) tokens per minute.")
                else: # request was made too close to previous request and got rate limited, retry request after limit resets
                    logging.warning(f"Rate limit temporarily exceeded (only {input_tokens} input tokens), retrying request in 120 seconds.")
                    time.sleep(120)
                    return get_llm_relevant_actions(actions_data=actions_data, query_list=query_list, synopsis=synopsis, doc_type=doc_type, context=context)

            else:
                raise FatalAPIError(f"Unexpected resource exhaustion error. Error ID: {error_id}")
        
        elif e.code == 403: # permission denied
            error_reason = e.details["error"]["details"][0]["reason"]
            if error_reason == "SERVICE_DISABLED":
                raise FatalAPIError("Generative Language API not enabled for the Google Cloud project this API key is associated with, cannot make calls to LLM.")
                
        else:
            raise NonFatalAPIError()
    

def get_qus_from_file(qus_file):
    if not os.path.exists(qus_file):
        logging.info(f"Question file {qus_file} does not exist.")
        return []
    else:
        with open(qus_file, "r", encoding="utf-8") as f:
            try:
                qus_list = json.load(f)
                if not isinstance(qus_list, list):
                    raise RetrievalError(f"Expected JSON file to contain a list, but contained {type(qus_list)} instead: {qus_file}")
                else:
                    logging.info(f"Loaded questions from {qus_file}.")
                    return qus_list
            except json.JSONDecodeError as e:
                raise RetrievalError(f"Error reading json from file {qus_file}: {str(e)}.")


def write_qus_to_file(qus_list, qus_file):
    os.makedirs(os.path.dirname(qus_file), exist_ok=True)
    with open(qus_file, "w", encoding="utf-8") as f:
        logging.info(f"Writing questions to {qus_file}.")
        json.dump(qus_list, f, indent=2, ensure_ascii=False)


def append_qus_to_file(new_qus, qus_file):
    try:
        prev_qus = get_qus_from_file(qus_file=qus_file)
    except RetrievalError as e:
        raise FileWriteError(f"Could not read existing questions from {qus_file}, so cannot write new questions to this file. Error: {str(e)}")
    else:
        prev_qus.extend(new_qus)
        write_qus_to_file(prev_qus, qus_file=qus_file)


def get_n_unique_qus(qu_dicts, n=10):
    n_unique_dicts = []
    remaining_qu_dicts = []
    processed = 0
    for i, q_dict in enumerate(qu_dicts):
        if len(n_unique_dicts) >= n:
            break
        query = q_dict["question"]
        current_queries = [d["question"] for d in n_unique_dicts]
        if query in current_queries:
            remaining_qu_dicts.append(q_dict)
        else:
            n_unique_dicts.append(q_dict)
        processed += 1
    unprocessed_dicts = qu_dicts[processed:]# is empty if len(qu_dicts) < n
    remaining_qu_dicts.extend(unprocessed_dicts)
    return n_unique_dicts, remaining_qu_dicts


def get_unique_batches(qu_dicts, batch_size=10):
    ###
    # Returns a list of batches of qu_dicts. Each batch is at most batch_size in length. All elements of qu_dicts exist in a batch.
    ###
    unique_batches = []

    # get_n_unique_qus implementation does not modify the qu_dicts argument,
    # but if in future this changes, need to take a deep copy of unprocessed_dicts here before passing it as an argument to get_n_unique_dicts.
    unprocessed_dicts = qu_dicts
    while unprocessed_dicts:
        new_batch, unprocessed_dicts = get_n_unique_qus(qu_dicts=unprocessed_dicts, n=batch_size)
        unique_batches.append(new_batch)
    return unique_batches   


def process_qus_in_synopsis(synopsis, context : FilterContext):
    doc_type = "bg_km"
    no_gaps_synopsis = "".join(synopsis.split())
    base_dir = context.get_qu_source_dir()
    qus_dir = os.path.join(base_dir, "untested")
    file_name = f"{doc_type}_{no_gaps_synopsis}_qus.json"
    qus_file = os.path.join(qus_dir, file_name)

    qus_full_details_list = get_qus_from_file(qus_file)
    logging.info(f"Loaded {len(qus_full_details_list)} questions for synopsis {synopsis}.")
    if qus_full_details_list == []:
        logging.info(f"No unprocessed questions found for synopsis {synopsis}, not processing this synopsis.")
        return
    all_batches = get_unique_batches(qu_dicts=qus_full_details_list, batch_size=10)
    # here we can assert that all the queries in each batch are unique.

    try:
        actions_data = get_synopsis_data_as_str(synopsis=synopsis, doc_type=doc_type)
    except RetrievalError as e:
        logging.warning(f"Could not retrieve action content for synopsis {synopsis}, not processing this synopsis.")
        return

    untested_qus = []
    passed_qus = []
    failed_qus = []

    for batch_num, qu_dicts_batch in enumerate(all_batches):
        # Indexing dicts by query - i.e. we require the queries in a batch to be unique.
        queries_batch = [qu_dict["question"] for qu_dict in qu_dicts_batch]
        qu_dicts_batch_query_indexed = {qu_dict["question"]: qu_dict for qu_dict in qu_dicts_batch}
        stored_ids_batch_query_indexed = {qus_full_details["question"]: qus_full_details["all_relevant_action_ids"] for qus_full_details in qu_dicts_batch}

        if context.get_current_calls() < context.get_max_calls():
            logging.info("Making API call for question batch.")
            try:
                gen_responses_batch = get_llm_relevant_actions(actions_data=actions_data, query_list=queries_batch, synopsis=synopsis, context=context, doc_type=doc_type)
            except APIError as e:
                logging.error(f"API error while processing questions for synopsis {synopsis}: {str(e)} Skipping remaining questions for this synopsis.")
                for i in range(batch_num, len(all_batches)):
                    untested_qus.extend(all_batches[i])
                break
            else:
                for response in gen_responses_batch:
                    query = response["query"]
                    stored_ids_for_query = stored_ids_batch_query_indexed.get(query, [])
                    gen_ids_for_query = response["relevant_action_ids"]
                    current_qu_dict = qu_dicts_batch_query_indexed[query]
                    current_qu_dict.update({"regenerated_ids":gen_ids_for_query})
                    
                    if set(gen_ids_for_query) <= set(stored_ids_for_query):
                        passed_qus.append(current_qu_dict)
                    else:
                        failed_qus.append(current_qu_dict)
        else:
            logging.info(f"User-set MAX_CALLS limit reached, skipping filtering remaining questions in synopsis {synopsis}.")
            for i in range(batch_num, len(all_batches)):
                untested_qus.extend(all_batches[i])
            break

    untested = len(untested_qus)
    logging.info(f"Total {len(qus_full_details_list)} questions for synopsis {synopsis}: {len(passed_qus)} passed, {len(failed_qus)} failed, {untested} untested.")

    pass_dir = os.path.join(base_dir, "passed")
    fail_dir = os.path.join(base_dir, "failed")
    untested_dir = os.path.join(base_dir, "untested")

    passed_file = os.path.join(pass_dir, file_name)
    failed_file = os.path.join(fail_dir, file_name)
    untested_file = os.path.join(untested_dir, file_name)

    try:
        append_qus_to_file(passed_qus, passed_file)
        logging.info(f"Added new set of questions that PASSED the filter to {passed_file}.")
    except FileWriteError as e:
        logging.error(f"{str(e)}")
    try:
        append_qus_to_file(failed_qus, failed_file)
        logging.info(f"Added new set of questions that FAILED the filter to {failed_file}.")
    except FileWriteError as e:
        logging.error(f"{str(e)}")
        
    write_qus_to_file(untested_qus, untested_file)
    logging.info(f"Overwrote {untested_file} with untested questions.")

            
def process_all_synopses(context : FilterContext):

    synopses = []
    for entry in os.scandir("action_data/background_key_messages/bg_km_synopsis"):
        synopses.append(entry.name)
    
    for i in range(len(synopses)):
        synopsis = synopses[(i+0) % len(synopses)]
        if context.get_current_synopses() < context.get_max_synopses():
            logging.info(f"Processing synopsis: {synopsis}")
            context.inc_current_synopses()
            if context.get_current_calls() < context.get_max_calls():
                process_qus_in_synopsis(synopsis=synopsis, context=context)
            else:
                logging.info(f"User-set MAX_CALLS limit reached, skipping processing remaining synopses.")
                break
        else:
            logging.info(f"User-set MAX_SYNOPSES limit reached, skipping processing remaining synopses.")
            break


def main():
    # synopsis = "Amphibian Conservation"
    # queries = ["What are the most effective interventions for controlling invasive predators to protect native amphibian populations?"]
    # stored_action_ids = [["797", "798", "821", "822", "825", "826", "827", "828", "829", "830", "839"]]
    # print(get_llm_relevant_actions(query_list=queries, synopsis=synopsis))

    logging.basicConfig(filename="logfiles/qus_filter_by_all_relevant_actions.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # disable httpx logging
    logging.getLogger("httpx").setLevel(logging.WARNING)

    QU_SOURCE_DIR = "question_gen_data/bg_km_multi_action_data/bg_km_qus/answerable"
    MAX_CALLS = 20
    MAX_SYNOPSES = 24

    context = FilterContext(qu_source_dir=QU_SOURCE_DIR, max_calls=MAX_CALLS, max_synopses=MAX_SYNOPSES)

    logging.info("STARTING question filtering process.")
    process_all_synopses(context=context)
    logging.info("ENDED question filtering process.")



if __name__ == "__main__":
    main()