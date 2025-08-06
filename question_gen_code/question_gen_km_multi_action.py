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

logging.basicConfig(filename="logfiles/km_multi_question_gen.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

QU_OUT_DIR = "question_gen_data/km_multi_action_data/km_multi_action_gen_qus"
MAX_CALLS = 3
#STUDIO_CALL_COUNT = 6
#API_CALL_COUNT
    # 6 + 19

def get_synopsis_data(synopsis, data_for_qu_type):
    # options for data_for_qu_type: "answerable", "unanswerable"
    no_gaps_synopsis = "".join(synopsis.split())
    try:
        if data_for_qu_type == "answerable":
            synopsis_file_path = f"question_gen_data/km_multi_action_data/km_synopsis_filtered_concat/km_{no_gaps_synopsis}_filtered_concat.txt"
        else:
            synopsis_file_path = f"question_gen_data/km_multi_action_data/km_synopsis_unfiltered_concat/km_{no_gaps_synopsis}_concat.txt"

        with open(synopsis_file_path, "r", encoding="utf-8") as f:
            content = f.read()

        if content == "" and data_for_qu_type == "answerable":
            logging.info(f"Zero filtered action files remaining for synopsis {synopsis}, skipping question generation for {synopsis}.")
            success = False
        elif content == "" and data_for_qu_type == "unanswerable":
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



def get_llm_response(synopsis, content, qu_type):
    # options for qu_type are: "answerable", "unanswerable"

    if qu_type == "answerable":
        prompt = f"""{content}\n\n\n
        Above is a document containing conservation actions, their effectiveness and key messages relating to them. This has been gathered by hand and forms part of the Conservation Evidence living evidence database. Each action is prefixed with a numerical id.
        You will be generating synthetic questions for use in an experiment. Generate four questions which requires drawing on multiple actions from this database. Also include a short model answer using only the evidence database provided. Cite the action id in your model answers. Also include a list of the action ids used to generate that question - these can be, but do not have to be, exactly the same as the action ids used to generate the answer. Keep the questions short and similar in style to the following example questions but vary the species and actions:
        Which predatory fish species are most effective at controlling invasive crayfish populations in European freshwater systems?
        Which conservation intervention can reduce the negative effects of artificial light pollution on nocturnal bat species' activity patterns?
        What actions are most effective for mitigating the impacts of offshore wind farms for biodiversity?
        What are the most beneficial actions I can take on agricultural land to improve impacts on pollinators and other aspects of biodiversity?
        Potential question types:
        a) Effects/outcomes of specific conservation actions
        b) Actions to achieve specific conservation objectives/outcomes
        c) Alternatives to achieve a particular conservation goal
        d) Trade-offs between different conservation actions
        e) Geographical and contextual variations in action effectiveness
        f) Scaling of actions and their impacts
        g) Timeframes for expected outcomes
        h) Factors associated with success or failure of conservation efforts
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
        logging.error(f"Server-side error, retrying request in 20 secs: {str(e)}.")
        time.sleep(20)
        return get_llm_response(synopsis=synopsis, content=content)

    except errors.ServerError as e:
        logging.error(f"Server side error, retrying request in 20 secs: {str(e)}")
        time.sleep(20)
        return get_llm_response(synopsis=synopsis, content=content)

#gen_questions = response.candidates[0].content.parts[0].text
#print("\n\n\n\nRESPONSE TEXT\n\n")
#print(response.text)



def get_prev_qus(synopsis, qu_type):
    # options for qu_type are: "answerable", "unanswerable"
    no_gaps_synopsis = "".join(synopsis.split())
    if qu_type == "answerable":
        outfile = os.path.join(QU_OUT_DIR, "answerable", f"km_{no_gaps_synopsis}_qus.json")
    elif qu_type == "unanswerable":
        outfile = os.path.join(QU_OUT_DIR, "unanswerable", f"km_{no_gaps_synopsis}_qus.json")
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
        outfile = os.path.join(QU_OUT_DIR, "answerable", f"km_{no_gaps_synopsis}_qus.json")
    elif qu_type == "unanswerable":
        outfile = os.path.join(QU_OUT_DIR, "unanswerable", f"km_{no_gaps_synopsis}_qus.json")
    else:
        logging.warning(f"Invalid argument {qu_type} given to parameter 'qu_type' in function 'write_all_qus'. File write failed.")
        return

    with open(outfile, 'w', encoding="utf-8") as outfile:
        json.dump(qus_list, outfile, indent=2)
    logging.info(f"Updated {outfile.name} with new questions.")



def process_all_synopses(qu_type):
    # options for qu_type: "answerable", "unanswerable"
    synopses = []
    for entry in os.scandir("action_data/key_messages/km_synopsis"):
        synopses.append(entry.name)
    num_synopses = len(synopses)
    
    call_count = 0
    for i in range(MAX_CALLS):
        synopsis = synopses[((i+21) % num_synopses)]
        content_retrieval_success, content = get_synopsis_data(synopsis, data_for_qu_type=qu_type)
        
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
        process_all_synopses(qu_type="unanswerable")
        logging.info("ENDED question generation process")
    except KeyboardInterrupt as e:
        logging.error(f"Keyboard interrupt: {str(e)}")
        logging.info("ENDED question generation process")




if __name__ == "__main__":
    main()

