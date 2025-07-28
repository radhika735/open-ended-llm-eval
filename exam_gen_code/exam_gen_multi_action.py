import os
from dotenv import load_dotenv
from google import genai 
from google.genai import types
from google.api_core import exceptions
import logging
from pydantic import BaseModel
import json
import time

load_dotenv()

logging.basicConfig(filename="logfiles/multi_exam_gen.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# disable httpx logging
logging.getLogger("httpx").setLevel(logging.WARNING)

QU_OUT_DIR = "exam_gen_data/multi_action_data"
MAX_CALLS = 24
#STUDIO_CALL_COUNT = 6
#API_CALL_COUNT = 6 or 7?
    # in loop faied +1
    # in loop succeeded +123?


def get_synopsis_data(synopsis):
    no_gaps_synopsis = "".join(synopsis.split())
    try:
        synopsis_file_path = f"exam_gen_data/km_synopsis_filtered_concat/km_{no_gaps_synopsis}_filtered_concat.txt"
        with open(synopsis_file_path, "r", encoding="utf-8") as f:
            content = f.read()
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



def get_llm_response(synopsis, content):
    rate_limited = False

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
        return rate_limited, new_qus
    except KeyError as e:
        logging.error(f"Unexpected API response format: {str(e)}.")
        return rate_limited, []
    except exceptions.ResourceExhausted as e:
        logging.error(f"Rate limit exceeded: {str(e)}.")
        rate_limited = True
        return rate_limited, []
    except exceptions.InternalServerError as e:
        logging.error(f"Server-side error, retrying request in 20secs: {str(e)}.")
        time.sleep(20)
        return get_llm_response(synopsis, content)


#gen_questions = response.candidates[0].content.parts[0].text
#print("\n\n\n\nRESPONSE TEXT\n\n")
#print(response.text)



def get_prev_qus(synopsis):
    no_gaps_synopsis = "".join(synopsis.split())
    outfile = os.path.join(QU_OUT_DIR, f"km_{no_gaps_synopsis}_qus.json")
    if os.path.exists(outfile):
        with open(outfile, "r", encoding="utf-8") as file:
            try:
                qus_list = json.load(file)
                logging.info(f"Loaded existing questions from {outfile}.")
            except json.JSONDecodeError:
                qus_list = []
                logging.error(f"Failed to load existing questions from {outfile}. Overwriting file.")
    else:
        qus_list = []
        logging.info(f"Creating new question output file {outfile}.")

    if not isinstance(qus_list, list):
        raise ValueError("Expected JSON file to contain a list")
    
    return qus_list



def write_all_qus(synopsis, qus_list):
    no_gaps_synopsis = "".join(synopsis.split())
    outfile = os.path.join(QU_OUT_DIR, f"km_{no_gaps_synopsis}_qus.json")

    with open(outfile, 'w', encoding="utf-8") as outfile:
        json.dump(qus_list, outfile, indent=2)
    logging.info(f"Updated {outfile.name} with new questions.")



def process_all_synopses():
    synopses = []
    for entry in os.scandir("action_data/key_messages/km_synopsis"):
        synopses.append(entry.name)
    num_synopses = len(synopses)
    
    call_count = 0
    for i in range(MAX_CALLS):
        synopsis = synopses[((i+11) % num_synopses)]
        success, content = get_synopsis_data(synopsis)
        
        if success:
            start = time.monotonic()
            rate_limited, new_qus = get_llm_response(synopsis=synopsis, content=content)
            call_count += 1
            if rate_limited:
                logging.error(f"{call_count} calls to API made before rate limit exceeded.")
                return
            else:
                logging.info(f"Generated {len(new_qus)} questions for synopsis {synopsis} in {(time.monotonic() - start):.3f} seconds.")
                
                qus_list = get_prev_qus(synopsis=synopsis)
                qus_list.extend(new_qus)
                write_all_qus(synopsis=synopsis, qus_list=qus_list)



def main():
    logging.info("STARTING question generation process.")
    process_all_synopses()
    logging.info("ENDED question generation process")



if __name__ == "__main__":
    main()

