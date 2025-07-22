import os
from dotenv import load_dotenv
import requests
import logging
import time
import json
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

logging.basicConfig(filename = "logfiles/exam_gen.log", level=logging.DEBUG)

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


def gen_exam_qus(action_nums : list, action_contents : str):
    ### V3 - ALTERING V2 TO RETURN MULTIPLE Q&A PAIRS:
    sys_msg = f"""
        Generate 5 difficult multi-form exam questions based on the provided conservation action information. 
        Also generate an example answer for each question, based on information only available in the provided context. The answers should be roughly 3 sentences in length.
        Additionally, for each question-answer pair, provide a brief proof of correctness (maximum 3 sentences) that includes reasons why that answer is correct based on the information on the action page.
        Follow these guidelines:

        1. Include a mix of question types. These may include:
            a) Effects/outcomes of a specific conservation action
            b) Actions to achieve specific conservation objectives/outcomes 
            c) Alternatives to achieve a particular conservation goal
            d) Trade-offs between different conservation actions
            e) Geographical and contextual variations in action effectiveness
            f) Scaling of actions and their impacts
            g) Timeframes for expected outcomes
            h) Factors associated with success or failure of conservation efforts

        Format each question as a JSON object:
        {{
        "question":"...",
        "source_actions": {action_nums},
        "example_answer":"...",
        "proof_of_correctness":"...",
        }}
        
        Ensure that once you have generated these questions, you recheck them for clarity, correctness and difficulty (these should not have necessarily obvious answers - use the source material). 
        Output these questions as a valid JSON array of question objects, starting with '[' and ending with ']'. 

    """


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

    ### V1 - BASED ON RADHIKA I'S SYS MSG:
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
        "model": "deepseek/deepseek-r1-0528:free",
        "max_tokens": 2048, 
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
            #logging.warning("Rate limit reached")
            time.sleep(60)  
            return gen_exam_qus(action_nums, action_contents)
        if response.status_code == 529:
            #logging.warning(f"Server error: {str(e)} - retrying request.")
            time.sleep(2)
            return gen_exam_qus(action_nums, action_contents)
        #logging.error(f"Error in API call: {str(e)}")
        return ""
    

def parse_json():
    pass


def process_files(file_list):
    try:
        file_contents = ""
        for file_path in file_list:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                file_contents = file_contents + "### CONTENT: ###" + content
        
        action_nums = []
        for file_path in file_list:
            action_nums.append(os.path.basename(file_path).split('_')[1])
        questions = gen_exam_qus(action_nums, content)
        print(questions)
    
    except FileNotFoundError:
        #logging.error(f"File not found: {file_path}")
        print(f"File not found: {file_path}")

        
def get_related_actions():
    pass


def main():
    #logging.info("Starting exam question generation process")
    process_files(["action_data/original/cleaned_textfiles/action_1_clean.txt"])


if __name__ == "__main__":
    main()