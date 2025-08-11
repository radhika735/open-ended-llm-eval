import json
from dotenv import load_dotenv
from google import genai 
from google.genai import types, errors
from google.api_core import exceptions
import logging
from pydantic import BaseModel
import os
from itertools import batched
import time
from question_gen_bg_km_multi_action import get_synopsis_data


logging.basicConfig(filename="logfiles/all_relevant_actions_filter.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

QU_SOURCE_DIR = "question_gen_data/bg_km_multi_action_data/bg_km_multi_action_gen_qus/answerable"
MAX_CALLS = 30
MAX_SYNOPSES = 1



class RelevantActions(BaseModel):
    query: str
    relevant_action_ids: list[str]



def get_llm_relevant_actions(query_list, synopsis):
    actions_content = get_synopsis_data(synopsis=synopsis, use_filtered_synopsis=False)

    prompt = f"""{actions_content}\n\n\n
    Above is a document containing conservation actions, their effectiveness and key messages relating to them. This has been gathered by hand and forms part of the Conservation Evidence living evidence database. Each action is prefixed with a numerical id.
    You are given a list of queries, each of which are answerable by using multiple actions in the document above. Your task is to return a list for each query, with each list containing the ids of ALL the actions that are relevant to that query. 
    Format your response as a list of JSON objects, with each object containing the query and the corresponding list of relevant action ids you have identified.

    Here are the queries: {query_list}
    """
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=8192),
                response_mime_type="application/json",
                response_schema=list[RelevantActions]
            )
        )

        ids = [relevant_actions_obj.model_dump() for relevant_actions_obj in response.parsed]
        success = True
        rate_limited = False
        return success, rate_limited, ids

    except KeyError as e:
        logging.error(f"Unexpected API response format: {str(e)}.")
        success = False
        rate_limited = False
        return success, rate_limited, [{"query":query, "relevant_action_ids":[]} for query in query_list]
    
    except exceptions.ResourceExhausted as e:
        logging.error(f"Rate limit exceeded: {str(e)}.")
        success = False
        rate_limited = True
        return success, rate_limited, []
    
    except exceptions.InternalServerError as e:
        logging.error(f"Server-side error, retrying request in 20 secs: {str(e)}.")
        time.sleep(20)
        return get_llm_relevant_actions(query_list=query_list, synopsis=synopsis)

    except errors.ServerError as e:
        logging.error(f"Server side error, retrying request in 20 secs: {str(e)}")
        time.sleep(20)
        return get_llm_relevant_actions(query_list=query_list, synopsis=synopsis)
    
    except TypeError as e:
        logging.error(f"Type error in API response: {str(e)}. Response content: {response.text if response else 'No response'}. Retrying request in 30 secs.")
        time.sleep(30)
        return get_llm_relevant_actions(query_list=query_list, synopsis=synopsis)



def get_qus_from_file(qus_file):
    if not os.path.exists(qus_file):
        logging.error(f"Question file {qus_file} does not exist.")
        return []

    with open(qus_file, "r", encoding="utf-8") as f:
        logging.info(f"Loading questions from {qus_file}.")
        qus_list = json.load(f)

    if not isinstance(qus_list, list):
        logging.error(f"Expected a list of questions in {qus_file}, but got {type(qus_list)}.")
        return []

    return qus_list



def write_qus_to_file(qus_list, qus_file):
    os.makedirs(os.path.dirname(qus_file), exist_ok=True)
    with open(qus_file, "w", encoding="utf-8") as f:
        logging.info(f"Writing questions to {qus_file}.")
        json.dump(qus_list, f, indent=2, ensure_ascii=False)



def process_qus_in_synopsis(synopsis):
    global MAX_CALLS

    no_gaps_synopsis = "".join(synopsis.split())

    base_dir = QU_SOURCE_DIR
    file_name = f"bg_km_{no_gaps_synopsis}_qus.json"

    qus_file = os.path.join(base_dir, file_name)
    qus_full_details_list = get_qus_from_file(qus_file)
    logging.info(f"Loaded {len(qus_full_details_list)} questions for synopsis {synopsis}.")

    passed_qus = []
    failed_qus = []

    for qus_full_details_batch in batched(qus_full_details_list, 10):
        
        qus_batch = [qus_full_details["question"] for qus_full_details in qus_full_details_batch]
        qus_full_details_batch_query_indexed = {qus_full_details["question"]: qus_full_details for qus_full_details in qus_full_details_batch}
        stored_ids_batch_query_indexed = {qus_full_details["question"]: qus_full_details["all_relevant_action_ids"] for qus_full_details in qus_full_details_batch}
        
        if MAX_CALLS >= 1:
            logging.info("Making API call for question batch.")
            api_call_success, rate_limited, gen_responses_batch = get_llm_relevant_actions(query_list=qus_batch, synopsis=synopsis)
            MAX_CALLS -= 1
        else:
            logging.info(f"User-set MAX_CALLS limit reached, skipping processing remaining questions in synopsis {synopsis}.")
            break
        
        if api_call_success:
            for response in gen_responses_batch:
                query = response["query"]
                stored_ids_for_query = stored_ids_batch_query_indexed.get(query, [])
                gen_ids_for_query = response["relevant_action_ids"]
                
                if stored_ids_for_query == gen_ids_for_query:
                    passed_qus.append(qus_full_details_batch_query_indexed[query])
                else:
                    failed_qus.append(qus_full_details_batch_query_indexed[query])
        else:
            break

    unprocessed = len(qus_full_details_list) - len(passed_qus) - len(failed_qus)
    logging.info(f"Total {len(qus_full_details_list)} questions for synopsis {synopsis}: {len(passed_qus)} passed, {len(failed_qus)} failed, {unprocessed} unprocessed.")

    pass_dir = os.path.join(base_dir, "passed")
    fail_dir = os.path.join(base_dir, "failed")

    os.makedirs(pass_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)
    passed_file = os.path.join(pass_dir, file_name)
    failed_file = os.path.join(fail_dir, file_name)

    logging.info(f"Writing passed questions to {passed_file}.")
    write_qus_to_file(passed_qus, passed_file)
    logging.info(f"Writing failed questions to {failed_file}.")
    write_qus_to_file(failed_qus, failed_file)


            
def process_all_synopses():
    global MAX_CALLS, MAX_SYNOPSES

    synopses = []
    for entry in os.scandir("action_data/background_key_messages/bg_km_synopsis"):
        synopses.append(entry.name)
    
    for s in synopses:
        if MAX_SYNOPSES >= 1:
            logging.info(f"Processing synopsis: {s}")
            MAX_SYNOPSES -= 1
            if MAX_CALLS >= 1:
                process_qus_in_synopsis(synopsis=s)
                MAX_CALLS -= 1
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
    logging.info("STARTING question filtering process.")
    process_all_synopses()
    logging.info("ENDED question filtering process.")



if __name__ == "__main__":
    main()