import os
from dotenv import load_dotenv
from google import genai 
from google.genai import types, errors
import logging
from pydantic import BaseModel
import json
import time

from utils.gen_qus_statistics import get_n_representative_qus_for_synopsis
from utils.action_retrieval import get_synopsis_data_as_str


class QuGenContext():
    def __init__(self, qu_out_dir, max_calls, doc_type="bg_km", prev_qus_dirs=[]):
        self.__qu_out_dir = qu_out_dir
        self.__prev_qus_dirs = prev_qus_dirs
        self.__max_calls = max_calls
        self.__current_calls = 0
        self.__doc_type = doc_type

    def get_max_calls(self):
        return self.__max_calls
    
    def get_current_calls(self):
        return self.__current_calls
    
    def inc_current_calls(self):
        self.__current_calls += 1
    
    def get_qu_out_dir(self):
        return self.__qu_out_dir

    def get_doc_type(self):
        return self.__doc_type
    
    def get_prev_qus_dirs(self):
        prev_qus_dirs_copy = self.__prev_qus_dirs.copy()
        return prev_qus_dirs_copy


def read_json_file(filename):
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as file:
            try:
                content = json.load(file)
                logging.info(f"Loaded existing questions from {filename}.")

                if not isinstance(content, list):
                    raise ValueError(f"Expected JSON file to contain a list, but contained {type(content)} instead.")
                else:
                    error = False
                    return error, content
            
            except json.JSONDecodeError as e:
                logging.warning(f"Failed to load existing questions from {filename}: {str(e)}.")
                error = True
                return error, []
    else:
        logging.info(f"File not found to read from: {filename}.")
        error = False
        return error, []


def write_to_json_file(filename, qus):
    if not os.path.exists(filename):
        logging.info(f"Creating new question output file {filename}.")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(qus, file, ensure_ascii=False, indent=2)
        

def append_to_json_file(filename, new_qus):
    error, prev_qus = read_json_file(filename=filename)
    if error == True:
        logging.warning(f"Loading existing questions failed, overwriting {filename} with new questions.")
    prev_qus.extend(new_qus)
    write_to_json_file(filename=filename, qus=prev_qus)
    logging.info(f"Updated {filename} with new questions.")


def append_qus(qus, synopsis, qu_type, qu_out_dir, doc_type="bg_km"):
    # qu_type can be "answerable" or "unanswerable"
    # doc_type can be "km" or "bg_km"
    no_gaps_synopsis = "".join(synopsis.split())

    if qu_type == "answerable":
        all_file = os.path.join(qu_out_dir, "answerable", "all", f"{doc_type}_{no_gaps_synopsis}_qus.json")
        append_to_json_file(filename=all_file, new_qus=qus)

        test_file = os.path.join(qu_out_dir, "answerable", "untested", f"{doc_type}_{no_gaps_synopsis}_qus.json")
        append_to_json_file(filename=test_file, new_qus=qus)

    elif qu_type == "unanswerable":
        filename = os.path.join(qu_out_dir, "unanswerable", "all", f"{doc_type}_{no_gaps_synopsis}_qus.json")
        append_to_json_file(filename=filename, new_qus=qus)

    else:
        logging.warning(f"Invalid argument {qu_type} given to parameter 'qu_type' in function 'write_all_qus'. File write failed.")
        return


def get_prev_qus(prev_qu_dirs, synopsis, doc_type="bg_km", max=15):
    retrieval_success = True
    all_qus = []

    no_gaps_synopsis = "".join(synopsis.split())

    for dir in prev_qu_dirs:
        filename = os.path.join(dir, f"{doc_type}_{no_gaps_synopsis}_qus.json")
        error, qus_dicts = read_json_file(filename)
        
        if error == True:
            retrieval_success = False # If an error occurs during loading from AT LEAST one of the previous questions files, 
                                    # retrieval success flag is set to False and returned from this function.
        else:
            qus = [qu_dict["question"] for qu_dict in qus_dicts]
            all_qus.extend(qus)
    
    top_n_qus = get_n_representative_qus_for_synopsis(qus_list=all_qus, synopsis=synopsis, n=max)
    qus_str = "\n".join(top_n_qus)
    return retrieval_success, qus_str


class QuestionAnswer(BaseModel):
    question: str
    answer: str
    action_ids_used_for_question_generation: list[str]
    action_ids_used_in_model_answer: list[str]
    all_relevant_action_ids: list[str]


def get_llm_response(context : QuGenContext, synopsis, actions_data, qu_type, prev_qus, doc_type="bg_km"):
    # options for qu_type are: "answerable", "unanswerable"
    # RETURNS (success : bool, fatal_error : bool, gen_qus : list[Dict]) where:
        # success is True if questions were generated, False if question generation did not (successfully) occur for any reason.
        # fatal_error is True if a condition has occurred such that no more API calls should be made - program needs to be rerun (potentially with some external condition changed before rerun).
            # fatal_error is False if no such condition has occurred, and future API calls can be made.
        # gen_qus is a list of the generated questions. The internal question dicts have fields aligning with the QuestionAnswer class.
    logging.info(f"Making request to generate questions for synopsis {synopsis}.")
    if prev_qus:
        prev_qus_prompt = f"""
        Here are some questions already generated. For question variety, make sure your questions do not ask about the same things that any of these questions ask about:
        {prev_qus}
        """
    else:
        prev_qus_prompt = ""

    if qu_type == "answerable":
        prompt = f"""{actions_data}\n\n\n
        Above is a document containing conservation actions, their effectiveness and key messages relating to them. This has been gathered by hand and forms part of the Conservation Evidence living evidence database. Each action is prefixed with a numerical id.
        You will be generating synthetic questions for use in an experiment. Generate seven questions which require drawing on multiple actions from this database. Also include a short model answer using only the evidence database provided. Cite the action id in your model answers, and also separately include a list of all these action ids used in the model answer. Also include a list of the action ids used to generate that question, and a list of ids of ALL the actions that are relevant to the question.
        Keep the questions short and similar in style to the following example questions but vary the species and actions:
        Which predatory fish species are most effective at controlling invasive crayfish populations in European freshwater systems?
        Which conservation intervention can reduce the negative effects of artificial light pollution on nocturnal bat species' activity patterns?
        What actions are most effective for mitigating the impacts of offshore wind farms for biodiversity?
        What are the most beneficial actions I can take on agricultural land to improve impacts on pollinators and other aspects of biodiversity?
        {prev_qus_prompt}

        Here is a list of potential question types. Each of the generated questions should have a different question types:
        Questions may ask about:
        a) Available actions to achieve specific conservation objectives/outcomes, or to avoid specific conservation threats
        b) Best / worst actions (according to a metric e.g. "most effective") to achieve specific conservation objectives/outcomes, or to avoid specific threats.
        c) Effects/outcomes of a specific conservation action or a general type of conservation action
        d) The benefits or negative impacts of specific conservation actions
        e) Comparison between the effects/impacts of similar actions.
        f) Alternative actions (to a given action) to achieve a particular conservation goal / avoid a particular threat.
        g) Evidence review (asking about the existence, quantity, or summary of available knowledge known about a specific conservation outcome)
        h) Factors associated with success or failure of conservation efforts
        i) Geographical and contextual variations in action effectiveness
        j) Scaling of actions and their impacts
        k) Timeframes for expected outcomes

        Guidelines to follow when generating the questions:

        1) Do NOT generate questions that can be answered by ONLY referring to the background information sections of actions.
        2) You may generate questions comparing specific actions, but these actions must be very similar in nature and an intuitive comparison. Not all actions can be intuitively compared even if they achieve the same conservation goal. Consider actions similar and comparable if they involve carrying out similar procedures / activities, not necessarily if they have similar outcomes.
        This is an example of a GOOD comparison: "What are the effects of planting trees with plastic tree guards, vs planting them with additional fertiliser?".
        This is an example of a BAD comparison: "To restore amphibian populations, what is the evidence for the effectiveness of creating new ponds compared to engaging volunteers in habitat management?"
        3) Leave HALF of the questions general. Scope the other HALF of the questions with more specific context, where the context is the specific conditions, location, or source of impact that scopes the question. A question may include more than one.
        Source of Impact / Sector: The human activity causing the problem that needs mitigation.
        Examples: "agricultural production", "business impacts", "onshore wind farms", "offshore wind farms", "complex agricultural supply chains", "underwater noise", "powerlines"
        Geography: The specific location of interest.
        Examples: "[country Y]", "the Greater North Sea region", "in this part of the world"
        Habitat Type: The environment where the action takes place.
        Examples: "grassland", "chalk grassland", "woodland", "agricultural land", "marine"

        Generate questions about {synopsis}.
        """
    elif qu_type == "unanswerable":
        prompt = f"""{actions_data}\n\n\n
        Above is a document containing conservation actions, their effectiveness and key messages relating to them. This has been gathered by hand and forms part of the Conservation Evidence living evidence database. Each action is prefixed with a numerical id.
        You will be generating synthetic questions for use in an experiment. Generate four questions which CANNOT be answered by any of the actions in this database. Keep the questions short. Include a model answer to this question, explaining clearly why the question is unanswerable.
        Here are some example reasons why a particular question may be unanswerable:
        a) Answering the question requires drawing on actions for which there has been no supporting evidence.
        b) The question asks about a factor which is not recorded in the database (e.g. cost-effectiveness)
        c) None of the actions in the database are relevant enough to answer the question.
        d) The question is too broad / not relevant to the database.
        Generate questions about {synopsis}.
        """
    else:
        logging.warning(f"Invalid argument {qu_type} given to parameter 'qu_type' in function 'get_llm_response'.")
        success = False
        fatal_error = False
        return success, fatal_error, []

    try:
        client = genai.Client()
        logging.info("Making API call.")
        model_name = "gemini-2.5-pro"

        input_tokens = client.models.count_tokens(model=model_name, contents=prompt).total_tokens
        if input_tokens > 100000: # synopsis size (+) prompt) may exceed input token limit for request. Do not make generation request.
            logging.warning(f"Total prompt for {synopsis} {doc_type} is {input_tokens} tokens long, exceeding input limit of 125,000 (empirically 100,000) tokens per minute. Skipping question generation for this synopsis.")
            success = False
            fatal_error = False
            return success, fatal_error, []
        
        if context.get_current_calls() >= context.get_max_calls():
            logging.info("User-set MAX CALLS exceeded. Halting question generation.")
            success = False
            fatal_error = True
            return success, fatal_error, []

        context.inc_current_calls()
        response = client.models.generate_content(
            model=model_name, 
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=8192),
                response_mime_type="application/json",
                response_schema=list[QuestionAnswer]
            )
        )
        new_qus = [qu_ans_obj.model_dump() for qu_ans_obj in response.parsed]
        success = True
        fatal_error = False
        return success, fatal_error, new_qus
    
    except KeyError as e:
        logging.error(f"Unexpected API response format: {str(e)}.")
        success = False
        fatal_error = False
        return success, fatal_error, []
    
    except TypeError as e:
        logging.error(f"Type error in API response: {str(e)}. Response content: {response.text if response else 'No response'}. Retrying request in 60 seconds.")
        time.sleep(60)
        return get_llm_response(context=context, synopsis=synopsis, actions_data=actions_data, qu_type=qu_type, prev_qus=prev_qus, doc_type=doc_type)

    except errors.ServerError as e:
        logging.error(f"Server side error, retrying request in 60 secs: {str(e)}")
        time.sleep(60)
        return get_llm_response(context=context, synopsis=synopsis, actions_data=actions_data, qu_type=qu_type, prev_qus=prev_qus, doc_type=doc_type)
    
    except errors.ClientError as e:
        logging.error(f"Client side error: {str(e)}")
        success = False
        
        if e.code == 429: # resource exhausted error (rate limit exceeded).
            error_dict = e.details
            error_id = error_dict["error"]["details"][0]["violations"][0]["quotaId"]

            if error_id == "GenerateRequestsPerDayPerProjectPerModel-FreeTier":
                logging.error(f"Exceeded free tier quota of 50 requests per day. Cannot continue question generation for any synopsis.")
                fatal_error = True
                return success, fatal_error, []
            
            elif error_id == "GenerateContentInputTokensPerModelPerMinute-FreeTier":
                if input_tokens >= 100000: # synopsis size itself (may have) exceeded input token size limit for free tier gemini (125000 input tokens / min), can't generate questions.
                    # should not reach this branch - have added a check for this before making the API call.
                    logging.warning(f"Total prompt for {synopsis} {doc_type} is {input_tokens} tokens long, exceeding input limit of 125,000 (empirically 100,000) tokens per minute. Skipping question generation for this synopsis.")
                    fatal_error = False
                    return success, fatal_error, []
                else: # request was made too close to previous request and got rate limited, retry request after limit resets.
                    logging.warning(f"Rate limit temporarily exceeded (only {input_tokens} input tokens), retrying request in 120 seconds.")
                    time.sleep(120)
                    return get_llm_response(context=context, synopsis=synopsis, actions_data=actions_data, qu_type=qu_type, prev_qus=prev_qus, doc_type=doc_type)
                
            else:
                logging.error(f"Unexpected resource exhaustion error. Error ID: {error_id}")
                fatal_error = True
                return success, fatal_error, []
            
        elif e.code == 403: # permission denied
            error_reason = e.details["error"]["details"][0]["reason"]
            if error_reason == "SERVICE_DISABLED":
                logging.error("Need to enable Generative Language API service for this Google Cloud project. Cannot continue question generation.")
                fatal_error = True
                return success, fatal_error, []

        else:
            fatal_error = False
            return success, fatal_error, []
    
    

def process_all_synopses(context : QuGenContext, qu_type, first_synopsis="Amphibian Conservation"):
    # options for qu_type: "answerable", "unanswerable"
    doc_type = context.get_doc_type()

    synopses = []
    for entry in os.scandir("action_data/key_messages/km_synopsis"):
        synopses.append(entry.name)
    num_synopses = len(synopses)

    try:
        offset = synopses.index(first_synopsis)
    except ValueError as e:
        offset = 0

    iteration = -1
    while context.get_current_calls() < context.get_max_calls():
        iteration += 1
        synopsis = synopses[((iteration+offset) % num_synopses)]
        actions_retrieval_success, actions = get_synopsis_data_as_str(synopsis, doc_type=doc_type)
        prev_qus_retrieval_success, prev_qus = get_prev_qus(prev_qu_dirs=context.get_prev_qus_dirs(), synopsis=synopsis, doc_type=doc_type, max=30)
        
        if actions_retrieval_success:
            if not prev_qus_retrieval_success:
                logging.warning(f"Unable to load previously generated questions for synopsis {synopsis}. Existing questions are much more likely to be regenerated by LLM.")

            start = time.monotonic()
            api_call_success, fatal_api_call_error, new_qus = get_llm_response(context=context, synopsis=synopsis, actions_data=actions, qu_type=qu_type, prev_qus=prev_qus, doc_type=doc_type)
            if api_call_success:
                logging.info(f"Generated {len(new_qus)} {qu_type} questions for synopsis {synopsis} in {(time.monotonic() - start):.3f} seconds.")
                append_qus(qus=new_qus, synopsis=synopsis, qu_type=qu_type, doc_type=doc_type, qu_out_dir=context.get_qu_out_dir())
            else:
                if fatal_api_call_error:
                    logging.error(f"Quitting question generation due to fatal api call error.")
                    return
        else:
            logging.warning(f"Failed to retrieve action data for synopsis {synopsis}, skipping question generation for this synopsis.")


def main():
    load_dotenv()
    
    logging.basicConfig(filename="logfiles/question_gen_multi_action.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # disable httpx logging
    logging.getLogger("httpx").setLevel(logging.WARNING)

    QU_OUT_DIR = "question_gen_data/bg_km_multi_action_data/bg_km_qus"
    prev_qus_dirs = ["question_gen_data/bg_km_multi_action_data/bg_km_qus/unanswerable/all"]
    MAX_CALLS = 3

    context = QuGenContext(qu_out_dir=QU_OUT_DIR, max_calls=MAX_CALLS, doc_type="bg_km", prev_qus_dirs=prev_qus_dirs)
    ## GENERATING ALL THE QUESTIONS
    try:
        logging.info("STARTING question generation process.")
        process_all_synopses(qu_type="unanswerable", context=context, first_synopsis="Reptile Conservation")
        logging.info("ENDED question generation process")
    except KeyboardInterrupt as e:
        logging.error(f"Keyboard interrupt: {str(e)}")
        logging.info("ENDED question generation process")

    # # Testing Peatland Conservation None responses:
    # synopsis = "Amphibian Conservation"
    # logging.info(f"STARTING Testing {synopsis} LLM responses.")
    # _, synopsis_data = get_synopsis_data_as_str(synopsis, doc_type="bg_km")
    # api_call_success, rate_limited, new_qus = get_llm_response(context=context, synopsis=synopsis, actions_data=actions, qu_type="answerable", prev_qus="")
    # print(new_qus)
    # logging.info(f"Generated {len(new_qus)} answerable questions for synopsis {synopsis}."  )
    # logging.info(f"ENDED Testing {synopsis} LLM responses.")


if __name__ == "__main__":
    main()

