import os
from dotenv import load_dotenv
from google import genai 
from google.genai import types, errors
from google.api_core import exceptions
import logging
from pydantic import BaseModel
import json
import time

load_dotenv()

logging.basicConfig(filename="logfiles/bg_km_multi_question_gen.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

QU_OUT_DIR = "question_gen_data/bg_km_multi_action_data/bg_km_multi_action_gen_qus"
MAX_CALLS = 1

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



class QuestionAnswer(BaseModel):
    question: str
    answer: str
    action_ids_used_for_question_generation: list[str]
    action_ids_used_in_model_answer: list[str]
    all_relevant_action_ids: list[str]



def get_llm_response(synopsis, content, qu_type):
    # options for qu_type are: "answerable", "unanswerable"

    if qu_type == "answerable":
        prompt = f"""{content}\n\n\n
        Above is a document containing conservation actions, their effectiveness and key messages relating to them. This has been gathered by hand and forms part of the Conservation Evidence living evidence database. Each action is prefixed with a numerical id.
        You will be generating synthetic questions for use in an experiment. Generate seven questions which require drawing on multiple actions from this database. Also include a short model answer using only the evidence database provided. Cite the action id in your model answers, and also separately include a list of all these action ids used in the model answer. Also include a list of the action ids used to generate that question, and a list of ids of ALL the actions that are relevant to the question.
        Keep the questions short and similar in style to the following example questions but vary the species and actions:
        Which predatory fish species are most effective at controlling invasive crayfish populations in European freshwater systems?
        Which conservation intervention can reduce the negative effects of artificial light pollution on nocturnal bat species' activity patterns?
        What actions are most effective for mitigating the impacts of offshore wind farms for biodiversity?
        What are the most beneficial actions I can take on agricultural land to improve impacts on pollinators and other aspects of biodiversity?

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
        prompt = f"""{content}\n\n\n
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
                response_schema=list[QuestionAnswer],
                temperature=1,
                seed=100
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
        logging.error(f"Server-side error, retrying request in 20 secs: {str(e)}.")
        time.sleep(20)
        return get_llm_response(synopsis=synopsis, content=content)

    except errors.ServerError as e:
        logging.error(f"Server side error, retrying request in 20 secs: {str(e)}")
        time.sleep(20)
        return get_llm_response(synopsis=synopsis, content=content)
    
    except TypeError as e:
        logging.error(f"Type error in API response: {str(e)}. Response content: {response.text if response else 'No response'}. Retrying request in 30 seconds.")
        time.sleep(30)
        return get_llm_response(synopsis=synopsis, content=content, qu_type=qu_type)


#gen_questions = response.candidates[0].content.parts[0].text
#print("\n\n\n\nRESPONSE TEXT\n\n")
#print(response.text)



def get_prev_qus(synopsis, qu_type):
    # options for qu_type are: "answerable", "unanswerable"
    no_gaps_synopsis = "".join(synopsis.split())
    if qu_type == "answerable":
        outfile = os.path.join(QU_OUT_DIR, "answerable", f"bg_km_{no_gaps_synopsis}_qus.json")
    elif qu_type == "unanswerable":
        outfile = os.path.join(QU_OUT_DIR, "unanswerable", f"bg_km_{no_gaps_synopsis}_qus.json")
    else:
        logging.warning(f"Invalid argument {qu_type} given to parameter 'qu_type' in function 'get_prev_qus'.")
        return []


    if os.path.exists(outfile):
        with open(outfile, "r", encoding="utf-8") as file:
            try:
                qus_list = json.load(file)
                logging.info(f"Loaded existing questions from {outfile}.")
            except json.JSONDecodeError:
                qus_list = []
                logging.warning(f"Failed to load existing questions from {outfile}. Overwriting file.")
    else:
        qus_list = []
        logging.info(f"Creating new question output file {outfile}.")

    if not isinstance(qus_list, list):
        raise ValueError("Expected JSON file to contain a list")
    
    return qus_list



def write_all_qus(synopsis, qus_list, qu_type):
    # options for qu_type are: "answerable", "unanswerable"
    no_gaps_synopsis = "".join(synopsis.split())
    if qu_type == "answerable":
        outfile = os.path.join(QU_OUT_DIR, "answerable", f"bg_km_{no_gaps_synopsis}_qus.json")
    elif qu_type == "unanswerable":
        outfile = os.path.join(QU_OUT_DIR, "unanswerable", f"bg_km_{no_gaps_synopsis}_qus.json")
    else:
        logging.warning(f"Invalid argument {qu_type} given to parameter 'qu_type' in function 'write_all_qus'. File write failed.")
        return

    with open(outfile, 'w', encoding="utf-8") as outfile:
        json.dump(qus_list, outfile, indent=2)
    logging.info(f"Updated {outfile.name} with new questions.")



def process_all_synopses(qu_type, use_filtered_synopsis=False):
    # options for qu_type: "answerable", "unanswerable"
    synopses = []
    for entry in os.scandir("action_data/key_messages/km_synopsis"):
        synopses.append(entry.name)
    num_synopses = len(synopses)
    
    call_count = 0
    for i in range(MAX_CALLS):
        synopsis = synopses[((i+16) % num_synopses)]
        content_retrieval_success, content = get_synopsis_data(synopsis, use_filtered_synopsis=use_filtered_synopsis)
        
        if content_retrieval_success:
            start = time.monotonic()
            api_call_success, rate_limited, new_qus = get_llm_response(synopsis=synopsis, content=content, qu_type=qu_type)
            call_count += 1
            if api_call_success:
                logging.info(f"Generated {len(new_qus)} {qu_type} questions for synopsis {synopsis} in {(time.monotonic() - start):.3f} seconds.")
                
                qus_list = get_prev_qus(synopsis=synopsis, qu_type=qu_type)
                qus_list.extend(new_qus)
                write_all_qus(synopsis=synopsis, qus_list=qus_list, qu_type=qu_type)
            else:
                if rate_limited:
                    logging.error(f"{call_count} calls to API made before rate limit exceeded.")
                return



def main():
    try:
        logging.info("STARTING question generation process.")
        process_all_synopses(qu_type="answerable", use_filtered_synopsis=False)
        logging.info("ENDED question generation process")
    except KeyboardInterrupt as e:
        logging.error(f"Keyboard interrupt: {str(e)}")
        logging.info("ENDED question generation process")




if __name__ == "__main__":
    main()

