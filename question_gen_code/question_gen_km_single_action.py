import os
from dotenv import load_dotenv
import requests
import logging
import time
import json
import re
import shutil

logging.basicConfig(filename = "logfiles/question_gen_km_single_action.log", level=logging.DEBUG)

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# The following four constants should either all point to action files that have supporting evidence, 
# or all point to actions files that have no supporting evidence.
# i.e.:
    # CONFIG 1: have evidence
        #SOURCE_DIR = "question_gen_data/single_action_data/km_evid_failedextract"
        # TARGET_DIR = "question_gen_data/single_action_data/km_evid_target"
        # FAILED_EXTRACT_DIR = "question_gen_data/single_action_data/km_evid_failedextract"
        # QU_OUT_FILE_PATH = "generated_questions_km_evid.json"
    # CONFIG 2: have no evidence
        #SOURCE_DIR = "question_gen_data/single_action_data/km_noevid_failedextract"
        # TARGET_DIR = "question_gen_data/single_action_data/km_noevid_target"
        # FAILED_EXTRACT_DIR = "question_gen_data/single_action_data/km_noevid_failedextract"
        # QU_OUT_FILE_PATH = "generated_questions_km_noevid.json"
SOURCE_DIR = "question_gen_data/single_action_data/km_evid_source"
TARGET_DIR = "question_gen_data/single_action_data/km_evid_target"
FAILED_EXTRACT_DIR = "question_gen_data/single_action_data/km_evid_failedextract"# Directory to put action files for which json parsing of generated questions failed.
QU_OUT_FILE_PATH = "generated_questions_km_evid.json"# Filename prefixed with prompt version numbers for different runs.

QUS_PER_CALL = 5 # Number of questions to generate per action file / set of action files given to the LLM.
NUM_CALLS = 1 # Maximum number of action files / sets of action files to generate questions for.


### Functions to parse output from LLM - this should be a string which has a list of JSON objects,
    # but could be a single JSON object, or could be incorrectly formatted.
def parse(json_str):
    try:
        parsed = json.loads(json_str)
        if isinstance(parsed, list):
            return parsed
        
        elif isinstance(parsed, dict):# Last year perhaps LLM tended to output as single element dict of "questions":..., I don't think this generally occurs with currently in use deepseek r1 responses.
            return parsed.get('questions', [parsed])
        
        return None
    
    except json.JSONDecodeError:
        return None


def extract_json(text):
    result = parse(text)

    if result is not None:
        return result

    json_pattern = r'\[[\s\S]*\]|\{[\s\S]*\}' #(S|\[(S(,S)*)?\])/i 
    match = re.search(json_pattern, text)

    if match:
        return parse(match.group())

    return None

###


def gen_llm_qus(action_nums : list, action_contents : str) -> str:
    # Below are multiple versions of the system message to generate the response. Topmost one is currently in use.

    ### V4 - ALTERING V3 TO HAVE QUESTIONS NOT REFERENCE SOURCE ACTION NUMS / SPECIFIC STUDIES, 
    # AND PHRASE QUS MORE GENERALLY RATHER THAN AS A TEST:
    sys_msg = f"""
        Generate {QUS_PER_CALL} difficult multi-form exam questions based on the provided conservation action information. 
        Also generate an example answer for each question, based on information only available in the provided context. The answers should be roughly 3 sentences in length.
        Additionally, for each question-answer pair, provide a brief proof of correctness (maximum 3 sentences) that includes reasons why that answer is correct based on the information on the action page.
        Follow these guidelines:

        1. Include one question that asks about:
            Which actions can achieve specific conservation objectives/outcomes (with the answer being the provided action itself).

        2. Attempt to include one question that asks about:
            Which are the MOST EFFECTIVE actions or the MOST COST-EFFECTIVE actions to achieve a particular conservation objective/outcome (with the answer being the provided action itself).

        3. The remaining questions should have a mix of question types. The question types should ask about aspects only from the following list:
            a) Effects/outcomes of the conservation action
            b) Which actions can achieve specific conservation objectives/outcomes (with the answer being the provided action itself)
            c) Alternatives to achieve a particular conservation goal
            d) Trade-offs between different conservation actions
            e) Geographical and contextual variations in action effectiveness
            f) Scaling of actions and their impacts
            g) Timeframes for expected outcomes
            h) Factors associated with success or failure of conservation efforts

        4. Ensure that the questions are clear, correct and appropriately difficult (they should not have necessarily obvious answers - use the source material). 
        
        5. Format each question as a JSON object:
            {{
            "question":"...",
            "source_actions": {','.join(action_nums)},
            "example_answer":"...",
            "proof_of_correctness":"...",
            }}
            Output these questions as a valid JSON array of question objects, starting with '[' and ending with ']'. 

        6. DO NOT reference the source action or specific studies in the questions. The question should be general, it needs to be phrased as if you are a research that does not have access to the provided context. 
        (e.g. DO NOT have questions like "Based on the information provided...").
           
    """


    ### V3 - ALTERING V2 TO RETURN MULTIPLE Q&A PAIRS:
    # sys_msg = f"""
    #     Generate {QUS_PER_CALL} difficult multi-form exam questions based on the provided conservation action information. 
    #     Also generate an example answer for each question, based on information only available in the provided context. The answers should be roughly 3 sentences in length.
    #     Additionally, for each question-answer pair, provide a brief proof of correctness (maximum 3 sentences) that includes reasons why that answer is correct based on the information on the action page.
    #     Follow these guidelines:

    #     1. Include a mix of question types. The question types should ask about aspects only from the following list, and at least one of the questions MUST ask about aspect (b) i.e. achieving a specific conservation goal using the provided action.:
    #         a) Effects/outcomes of the conservation action
    #         b) Which actions can achieve specific conservation objectives/outcomes (with the answer being the provided action itself)
    #         c) Alternatives to achieve a particular conservation goal
    #         d) Trade-offs between different conservation actions
    #         e) Geographical and contextual variations in action effectiveness
    #         f) Scaling of actions and their impacts
    #         g) Timeframes for expected outcomes
    #         h) Factors associated with success or failure of conservation efforts

    #     Format each question as a JSON object:
    #     {{
    #     "question":"...",
    #     "source_actions": {action_nums},
    #     "example_answer":"...",
    #     "proof_of_correctness":"...",
    #     }}
        
    #     Ensure that once you have generated these questions, you recheck them for clarity, correctness and difficulty (these should not have necessarily obvious answers - use the source material). 
    #     Output these questions as a valid JSON array of question objects, starting with '[' and ending with ']'. 

    # """


    ### V2 - ADDING PROOF OF CORRECTNESS TO V1:
    # sys_msg = f"""
    #     Generate one difficult multi-form exam question based on the provided conservation action information. 
    #     Also generate an example answer to the question, based on information only available in the provided context. The answer should be roughly 3 sentences in length.
    #     Additionally, provide a brief proof of correctness (maximum 3 sentences) that includes reasons why the provided answer is correct based on the information on the action page.

    #     The question should ask about one of the following aspects of the provided conservation action:
    #         a) Effects/outcomes of a specific conservation action
    #         b) Actions to achieve specific conservation objectives/outcomes 
    #         c) Alternatives to achieve a particular conservation goal
    #         d) Trade-offs between different conservation actions
    #         e) Geographical and contextual variations in action effectiveness
    #         f) Scaling of actions and their impacts
    #         g) Timeframes for expected outcomes
    #         h) Factors associated with success or failure of conservation efforts

    #     2. Pay attention to but do not explicitly ask questions about the effectiveness rating.
    #     3. Focus on conservation actions, outcomes, and their relationships. With actions referencing more than one study, see how they compare or differ. 
    #     4. Ask a question about the context provided, but DO NOT reference specific studies or use past tense (e.g. do not have the question sound like "Based on the information, provided...").

    #     Use Bloom's taxonomy (Remember, Create, Evaluate, Analyse, Apply, Understand) to evaluate which category the question belongs to. Use these and only these categories.
    #     Evaluate your assignment of Bloom's category AFTER generating each question. 

    #     Format the question as a JSON object:
    #     {{
    #     "question":"...",
    #     "source_actions": {action_nums},
    #     "example_answer":"...",
    #     "proof_of_correctness":"...",
    #     "bloom_level":"...",
    #     }}
        
    #     Ensure that once you have generated these questions, you recheck them for clarity, correctness and difficulty (these should not have necessarily obvious answers - use the source material).
    #     Output the question as a valid JSON object, starting with '{{' and ending with '}}'.

    # """


    ### V1 - BASED ON SYS MSG USED FOR MCQ GENERATION:
    # sys_msg = f"""
    #     Generate one difficult multi-form exam question based on the provided conservation action information. 
    #     Also generate an example answer to the question. The answer should be roughly 3 sentences in length.

    #     1. The question should ask about one of the following aspects of the provided conservation action:
    #         a) Effects/outcomes of a specific conservation action
    #         b) Actions to achieve specific conservation objectives/outcomes 
    #         c) Alternatives to achieve a particular conservation goal
    #         d) Trade-offs between different conservation actions
    #         e) Geographical and contextual variations in action effectiveness
    #         f) Scaling of actions and their impacts
    #         g) Timeframes for expected outcomes
    #         h) Factors associated with success or failure of conservation efforts

    #     2. Include questions about achieving desired outcomes (e.g., "How to increase the abundance of native bees?"). Pay attention to but do not explicitly ask questions about the effectiveness rating.
    #     3. Focus on conservation actions, outcomes, and their relationships. With actions referencing more than one study, see how they compare or differ. 
    #     4. Ask questions about the context provided, but DO NOT reference specific studies or use past tense (e.g. do not have questions like "Based on the information, provided...").
    #     5. Use Bloom's taxonomy (Remember, Create, Evaluate, Analyse, Apply, Understand) for question variety. Use these and only these category names.
    #     6. Provide documentation from the text itself (source) and a proof of correctness that includes reasons why the provided answer is correct based on the information on the action page.

    #     Evaluate your assignment of Bloom's category AFTER generating each question. 
    #     Format each question as a JSON object:
    #     {{
    #     "question": "...",
    #     "source_actions": {action_nums},
    #     "documentation": "...",
    #     "example_answer": "...",
    #     "proof_of_correctness": "...",
    #     "bloom_level": "..."
    #     }}
    #     Ensure that once you have generated these questions, you recheck them for clarity, correctness and difficulty (these should not have necessarily obvious answers - use the source material). You may remove questions as you deem appropriate, but you must return at least ONE. 
    #     Output these questions as a valid JSON array of question objects, starting with '[' and ending with ']'. 
    #     """

    data = {
        "model": "openai/o4-mini",# Can test multiple models and change. Prompt optimised for deepseek/deepseek-r1-0528:free responses.
        "max_tokens": 1500, 
        "messages": [
            {   
                "role":"system",
                "content": sys_msg
            },
            {
                "role": "user",
                "content": action_contents
            }
        ]
    }

    try:
        response = requests.post(
            url = "https://openrouter.ai/api/v1/chat/completions",
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json = data
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    except requests.exceptions.RequestException as e:
        print("Exception:", e)
        if response.status_code == 429:  
            logging.warning("Rate limit reached")
            time.sleep(60)  
            return gen_llm_qus(action_nums, action_contents)
        if response.status_code == 529:
            logging.warning(f"Server error: {str(e)} - retrying request.")
            time.sleep(2)
            return gen_llm_qus(action_nums, action_contents)
        logging.error(f"Error in API call: {str(e)}")
        return ""
    except KeyError as e:
        logging.error(f"Unexpected API response format: {str(e)}")
        print(response.json())
        return ""


### Takes a single action file, or a list of a few related action files and generates questions based on them.
    # Updates question_list with these generated questions and writes to the file at QU_OUT_FILE_PATH.
    # Returns True if questions were successfully generated, False otherwise.
    # Logs time taken for guestion generation and number of questions generated.
def process_files(file_list, question_list) -> bool:
    try:
        file_contents = ""
        for file_path in file_list:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                file_contents = file_contents + "### CONTENT: ###" + content
        
        action_nums = []
        for file_path in file_list:
            action_nums.append(os.path.basename(file_path).split('_')[1])

        start = time.monotonic()
        llm_response = gen_llm_qus(action_nums, content)
        logging.info(f"Generated qs for action(s) {action_nums} in {(time.monotonic() - start):.3f} seconds")
        questions = extract_json(llm_response)

        if questions:
            logging.info(f"Extracted {len(questions)} questions for the following actions: {action_nums}")
            question_list.extend(questions)
            
            with open(QU_OUT_FILE_PATH, 'w', encoding='utf-8') as outfile:
                json.dump(question_list, outfile, indent=2)
            
            logging.info(f"Updated {QU_OUT_FILE_PATH} with new questions")

            extraction_success = True 
        else:
            logging.warning(f"Failed to extract questions for the following actions: {action_nums}")
            print(llm_response)
            extraction_success = False

        return extraction_success

    except FileNotFoundError:
        logging.error(f"Action file not found: {file_path}")
        print(f"Action file not found: {file_path}")

        
def get_related_actions():
    pass


### Process all the action files in the SOURCE_DIR.
    # Processes each file and moves it to the TARGET_DIR if questions were successfully generated, or to the FAILED_EXTRACT_DIR if not.
    # Logs the number of files processed dynamically and the time taken to process the whole directory.
    # If reset_out_file is True, it resets the QU_OUT_FILE_PATH file to an empty list, 
    # otherwise the new questions that generated are appended to the existing list in QU_OUTPUT_FILE.
def process_action_dir(reset_out_file : bool):
    try:
        start_time = time.monotonic()
        if not os.path.exists(TARGET_DIR):
            os.makedirs(TARGET_DIR)
        
        if not reset_out_file and os.path.exists(QU_OUT_FILE_PATH):
            with open(QU_OUT_FILE_PATH, 'r', encoding='utf-8') as qu_out_file:
                question_list = json.load(qu_out_file)
            logging.info(f"Loaded existing questions from {QU_OUT_FILE_PATH}")
        else:
            question_list = []
            logging.info(f"Resetting question output file {QU_OUT_FILE_PATH}")
        
        files_processed = 0

        for entry in os.scandir(SOURCE_DIR):
            if files_processed >= NUM_CALLS:
                break
            
            extraction_success = process_files([entry.path], question_list)
            
            if extraction_success:
                shutil.move(entry.path, os.path.join(TARGET_DIR, entry.name))
                logging.debug(f"Moved {entry.name} to {TARGET_DIR}")
            else:
                shutil.move(entry.path, os.path.join(FAILED_EXTRACT_DIR, entry.name))
                logging.debug(f"Moved {entry.name} to {FAILED_EXTRACT_DIR} due to extraction failure")

            files_processed += 1
            logging.info(f"Processed {files_processed} files so far")
        
        end_time = time.monotonic()
        total_time = end_time - start_time
        
        logging.info(f"Exam generation completed. Processed {files_processed} files in {total_time:.2f} seconds")
        logging.info(f"Generated {len(question_list)} questions in total")

    except FileNotFoundError as e:
        logging.error(f"Source action directory not found: {e}")


def main():
    logging.info("Starting exam question generation process")
    
    process_action_dir(reset_out_file=False)
    


if __name__ == "__main__":
    main()