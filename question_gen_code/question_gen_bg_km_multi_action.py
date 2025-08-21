import os
from dotenv import load_dotenv
from google import genai 
from google.genai import types, errors
from google.api_core import exceptions
import logging
from pydantic import BaseModel
import json
import time

class Context():
    def __init__(self, qu_out_dir, max_calls, prev_qus_dirs=[]):
        self.__qu_out_dir = qu_out_dir
        self.__prev_qus_dirs = prev_qus_dirs
        self.__max_calls = max_calls

    def get_max_calls(self):
        return self.__max_calls
    
    def get_qu_out_dir(self):
        return self.__qu_out_dir
    
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



def append_qus(qus, synopsis, qu_type, context : Context):
    no_gaps_synopsis = "".join(synopsis.split())

    if qu_type == "answerable":
        all_file = os.path.join(context.get_qu_out_dir(), "answerable", "all", f"bg_km_{no_gaps_synopsis}_qus.json")
        append_to_json_file(filename=all_file, new_qus=qus)

        test_file = os.path.join(context.get_qu_out_dir(), "answerable", "untested", f"bg_km_{no_gaps_synopsis}_qus.json")
        append_to_json_file(filename=test_file, new_qus=qus)

    elif qu_type == "unanswerable":
        filename = os.path.join(context.get_qu_out_dir(), "unanswerable", f"bg_km_{no_gaps_synopsis}_qus.json")
        append_to_json_file(filename=filename, new_qus=qus)

    else:
        logging.warning(f"Invalid argument {qu_type} given to parameter 'qu_type' in function 'write_all_qus'. File write failed.")
        return
    


def get_synopsis_data(synopsis, use_filtered_synopsis=False):
    no_gaps_synopsis = "".join(synopsis.split())
    try:
        if use_filtered_synopsis:
            synopsis_file_path = f"question_gen_data/bg_km_multi_action_data/bg_km_synopsis_filtered_concat/bg_km_{no_gaps_synopsis}_filtered_concat.txt"
        else:
            synopsis_file_path = f"question_gen_data/bg_km_multi_action_data/bg_km_synopsis_unfiltered_concat/bg_km_{no_gaps_synopsis}_unfiltered_concat.txt"

        with open(synopsis_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if content == "" and use_filtered_synopsis:
            logging.info(f"Zero filtered action files remaining for synopsis {synopsis}, skipping question generation for {synopsis}.")
            success = False
        elif content == "" and use_filtered_synopsis == False:
            logging.error(f"No content in unfiltered action files for synopsis {synopsis} (see {synopsis_file_path}). Skipping question generation for {synopsis}.")
            success = False
        else:
            success = True

        return success, content
    
    except FileNotFoundError:
        logging.error(f"Actions for synopsis {synopsis} file not found: {synopsis_file_path}")
        success = False
        return success, ""



def get_prev_qus(context : Context, synopsis, max=20):
    retrieval_success = True
    all_qus = []

    no_gaps_synopsis = "".join(synopsis.split())

    for dir in context.get_prev_qus_dirs():
        filename = os.path.join(dir, f"bg_km_{no_gaps_synopsis}_qus.json")
        error, qus_dicts = read_json_file(filename)
        
        if error == True:
            retrieval_success = False # If an error occurs during loading from AT LEAST one of the previous questions files, 
                                    # retrieval success flag is set to False and returned from this function.
        else:
            qus = [qu_dict["question"] for qu_dict in qus_dicts]
            all_qus.extend(qus)
    
    qus_pruned = all_qus[:max]
    qus_str = "\n".join(qus_pruned)
    return retrieval_success, qus_str



class QuestionAnswer(BaseModel):
    question: str
    answer: str
    action_ids_used_for_question_generation: list[str]
    action_ids_used_in_model_answer: list[str]
    all_relevant_action_ids: list[str]



def get_llm_response(synopsis, actions_data, qu_type, prev_qus):
    # options for qu_type are: "answerable", "unanswerable"
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
        rate_limited = False
        return success, rate_limited, []

    try:
        client = genai.Client()
        logging.info(f"Making API call.")
        response = client.models.generate_content(
            model="gemini-2.5-pro", 
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=8192),
                response_mime_type="application/json",
                response_schema=list[QuestionAnswer]
            )
        )
        new_qus = [qu_ans_obj.model_dump() for qu_ans_obj in response.parsed]
        success = True
        rate_limited = False
        return success, rate_limited, new_qus
    
    except KeyError as e:
        logging.error(f"Unexpected API response format: {str(e)}.")
        success = False
        rate_limited = False
        return success, rate_limited, []
    
    except exceptions.ResourceExhausted as e:
        logging.error(f"Rate limit exceeded: {str(e)}.")
        success = False
        rate_limited = True
        return success, rate_limited, []
    
    except exceptions.InternalServerError as e:
        logging.error(f"Server-side error, retrying request in 60 secs: {str(e)}.")
        time.sleep(60)
        return get_llm_response(synopsis=synopsis, actions_data=actions_data, qu_type=qu_type, prev_qus=prev_qus)

    except errors.ServerError as e:
        logging.error(f"Server side error, retrying request in 60 secs: {str(e)}")
        time.sleep(60)
        return get_llm_response(synopsis=synopsis, actions_data=actions_data, qu_type=qu_type, prev_qus=prev_qus)
    
    except TypeError as e:
        logging.error(f"Type error in API response: {str(e)}. Response content: {response.text if response else 'No response'}. Retrying request in 60 seconds.")
        time.sleep(60)
        return get_llm_response(synopsis=synopsis, actions_data=actions_data, qu_type=qu_type, prev_qus=prev_qus)



def process_all_synopses(context : Context, qu_type, use_filtered_synopsis=False):
    # options for qu_type: "answerable", "unanswerable"
    synopses = []
    for entry in os.scandir("action_data/key_messages/km_synopsis"):
        synopses.append(entry.name)
    num_synopses = len(synopses)
    
    call_count = 0
    for i in range(context.get_max_calls()):
        synopsis = synopses[((i+16) % num_synopses)]
        actions_retrieval_success, actions = get_synopsis_data(synopsis, use_filtered_synopsis=use_filtered_synopsis)
        prev_qus_retrieval_success, prev_qus = get_prev_qus(context=context, synopsis=synopsis)
        
        if actions_retrieval_success:

            if not prev_qus_retrieval_success:
                logging.warning(f"Unable to load previously generated questions for synopsis {synopsis}. Existing questions are much more likely to be regenerated by LLM.")

            start = time.monotonic()
            api_call_success, rate_limited, new_qus = get_llm_response(synopsis=synopsis, actions_data=actions, qu_type=qu_type, prev_qus=prev_qus)
            call_count += 1
            if api_call_success:
                logging.info(f"Generated {len(new_qus)} {qu_type} questions for synopsis {synopsis} in {(time.monotonic() - start):.3f} seconds.")
                append_qus(qus=new_qus, synopsis=synopsis, qu_type=qu_type, context=context)
            else:
                if rate_limited:
                    logging.error(f"{call_count} calls to API made before rate limit exceeded.")
                return



def main():
    load_dotenv()
    
    logging.basicConfig(filename="logfiles/bg_km_multi_question_gen.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # disable httpx logging
    logging.getLogger("httpx").setLevel(logging.WARNING)

    QU_OUT_DIR = "question_gen_data/bg_km_multi_action_data/bg_km_qus"
    prev_qus_dirs = ["question_gen_data/bg_km_multi_action_data/bg_km_qus/answerable/all"]
    MAX_CALLS = 21

    context = Context(qu_out_dir=QU_OUT_DIR, max_calls=MAX_CALLS, prev_qus_dirs=prev_qus_dirs)

    ### GENERATING ALL THE QUESTIONS
    # try:
    #     logging.info("STARTING question generation process.")
    #     process_all_synopses(qu_type="answerable", use_filtered_synopsis=False, context=context)
    #     logging.info("ENDED question generation process")
    # except KeyboardInterrupt as e:
    #     logging.error(f"Keyboard interrupt: {str(e)}")
    #     logging.info("ENDED question generation process")

    ## Testing Peatland Conservation None responses:
    synopsis = "Amphibian Conservation"
    logging.info(f"STARTING Testing {synopsis} LLM responses.")
    _, synopsis_data = get_synopsis_data(synopsis, use_filtered_synopsis=False)
    api_call_success, rate_limited, new_qus = get_llm_response(synopsis=synopsis, actions_data=synopsis_data, qu_type="answerable", prev_qus="")
    print(new_qus)
    logging.info(f"Generated {len(new_qus)} answerable questions for synopsis {synopsis}."  )
    logging.info(f"ENDED Testing {synopsis} LLM responses.")


if __name__ == "__main__":
    main()

