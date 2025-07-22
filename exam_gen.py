import os
from dotenv import load_dotenv
import requests
import logging
import time
import json

logging.basicConfig(filename = "logfiles/exam_gen.log", level=logging.DEBUG)


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def gen_exam_qus(action_number, action_content):
    sys_msg = f"""
        Generate one difficult multi-form exam question based on the provided conservation action information. 
        Also generation an answer to the question. The answer should be roughly 3 sentences in length.

        1. The question should ask about one of the following aspects of the provided conservation action:
            a) Effects/outcomes of a specific conservation action
            b) Actions to achieve specific conservation objectives/outcomes 
            c) Alternatives to achieve a particular conservation goal
            d) Trade-offs between different conservation actions
            e) Geographical and contextual variations in action effectiveness
            f) Scaling of actions and their impacts
            g) Timeframes for expected outcomes
            h) Factors associated with success or failure of conservation efforts

        2. Include questions about achieving desired outcomes (e.g., "How to increase the abundance of native bees?"). Pay attention to but do not explicitly ask questions about the effectiveness rating.
        3. Focus on conservation actions, outcomes, and their relationships. With actions referencing more than one study, see how they compare or differ. 
        4. Ask questions about the context provided, but DO NOT reference specific studies or use past tense (e.g. do not have questions like "Based on the information, provided...").
        5. Use Bloom's taxonomy (Remember, Create, Evaluate, Analyse, Apply, Understand) for question variety. Use these and only these category names.
        6. Provide documentation from the text itself (source) and a proof of correctness that includes reasons why the provided answer is correct based on the information on the action page.

        Evaluate your assignment of Bloom's category AFTER generating each question. 
        Format each question as a JSON object:
        {{
        "question": "...",
        "source_action": {action_number},
        "documentation": "...",
        "example_answer": "...",
        "proof_of_correctness": "...",
        "bloom_level": "..."
        }}
        Ensure that once you have generated these questions, you recheck them for clarity, correctness and difficulty (these should not have necessarily obvious answers - use the source material). You may remove questions as you deem appropriate, but you must return at least ONE. 
        Output these questions as a valid JSON array of question objects, starting with '[' and ending with ']'. 
        """

    data = json.dumps({
        "model": "deepseek/deepseek-r1:free", ### CHANGE THIS
        "max_tokens": 2048, 
        "system": sys_msg,
        "messages": [
            {
                "role": "user",
                "content": action_content
            }
        ]
    })

    try:
        response = requests.post(
            url = "https://openrouter.ai/api/v1/chat/completions",
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            data = data
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        if response.status_code == 429:  
            #logging.warning("Rate limit reached")
            time.sleep(60)  
            return gen_exam_qus(action_number, action_content)
        if response.status_code == 529:
            #logging.warning(f"Server error: {str(e)} - retrying request.")
            time.sleep(2)
            return gen_exam_qus(action_number, action_content)
        #logging.error(f"Error in API call: {str(e)}")
        return ""
    

def process_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        action_num = os.path.basename(file_path).split('_')[1]
        questions = gen_exam_qus(action_num, content)
        print(questions)
    except FileNotFoundError:
        #logging.error(f"File not found: {file_path}")
        print(f"File not found: {file_path}")

        

def main():
    #logging.info("Starting exam question generation process")
    process_file("action_data/original/cleaned_textfiles/action_1_clean.txt")

if __name__ == "__main__":
    main()