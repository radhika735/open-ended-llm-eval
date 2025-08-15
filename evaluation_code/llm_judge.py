import os
from dotenv import load_dotenv
import logging
from google import genai 
from google.genai import types, errors
from google.api_core import exceptions
from pydantic import BaseModel

logging.basicConfig(level=logging.DEBUG, filename="logfiles/llm_judge.log")

load_dotenv()

# def get_client():
#     api_key = os.getenv("GEMINI_API_KEY")
#     if not api_key:
#         raise ValueError("GEMINI_API_KEY environment variable is required.")
#     return OpenAI(
#         base_url="https://openrouter.ai/api/v1",
#         api_key=api_key,
#     )


class Evaluation(BaseModel):
    accuracy_score : str
    score_reason : str
    action_ids_used_in_judgement : list[int]



def evaluate(question, answer, docs):
    prompt =  f"""{docs}

        Above is a document containing conservation actions, their effectiveness and key messages relating to them. This has been gathered by hand and forms part of the Conservation Evidence living evidence database. Each action is prefixed with a numerical id.
        You are given a question about conservation, and an answer to that question that a model has generated.
        Rate on a 5 point Likert scale (1 being poor and 5 being excellent) whether the answer has given the most relevant information first with respect to the Conservation Evidence database of documents given to you. Penalise answers that list less relevant information before more relevant information, and reward answers that list statements in the correct order: most strongly relevant to least strongly relevant to the question.
        Also rate on a 5 point Likert scale (1 being poor and 5 being excellent) whether the answer explains the research of why those answers are true.
        Also rate on a 5 point Likert scale (1 being poor and 5 being excellent) whether the answer is concise and does not include excessive detail.
        Take the sum of the scores you have assigned as the overall accuracy score of the answer.
        You must also include a concise (maximum 5 sentence) reason explaining why you have assigned that score. 
        Additionally you must include the IDs of the actions you use to make the judgement.

        Here is the question: {question}
        Here is the potential answer: {answer}
        Evaluate this answer.
    """

    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-2.5-pro", 
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=8192),
            response_mime_type="application/json",
            response_schema=Evaluation
        )
    )

    return response




if __name__ == "__main__":
    logging.info("STARTING answer evaluation process.")
    with open("question_gen_data/km_multi_action_data/km_synopsis_unfiltered_concat/km_BatConservation_concat.txt", 'r', encoding="utf-8") as f:
        docs = f.read()
    question = "What are the most effective interventions for reducing bat fatalities at wind turbines?"
    oracle_action_ids = []

    # Claude opus 4 answer:
    #answer = "Based on scientific evidence, the most effective interventions for reducing bat fatalities at wind turbines are:\n\n**1. Prevent turbine blades from turning at low wind speeds ('feathering') - Rated as Beneficial**\n- Five out of six studies in the USA and Canada found that feathering turbine blades resulted in fewer bat fatalities compared to conventionally operated turbines\n- This involves keeping blades stationary or nearly stationary during low wind conditions\n\n**2. Increase the wind speed at which turbines become operational ('cut-in speed') - Rated as Beneficial**\n- Ten out of 12 studies in the USA and Canada found that increasing cut-in speeds resulted in fewer bat fatalities\n- This prevents turbines from operating during periods when bats are most active (typically low wind conditions)\n- Often implemented alongside feathering for maximum effectiveness\n\n**3. Slow rotation of turbine blades at low wind speeds - Rated as Likely to be beneficial**\n- A replicated, randomized, controlled study in Canada found reduced bat fatalities when turbine blades were slowed at low wind speeds\n- This represents a middle ground between full operation and complete feathering\n\n**4. Automatically reduce turbine blade rotation when bat activity is high - Rated as Likely to be beneficial**\n- Two replicated studies in Germany and the USA found fewer bat fatalities when blade rotation was automatically reduced based on predicted bat activity\n- Uses real-time or predictive models to adjust turbine operation\n\n**5. Deter bats from turbines using ultrasound - Rated as Unknown effectiveness**\n- Mixed results from studies in the USA, with effectiveness varying by species and conditions\n- Some studies showed 21-64% reductions in fatalities, while others showed minimal or inconsistent effects\n- More research needed to determine optimal deployment\n\nThe most reliable and proven interventions are operational curtailment strategies (feathering and increasing cut-in speeds), which have shown consistent benefits across multiple studies. These methods work because most bat fatalities occur during low wind speeds when bats are most active."
    
    # Gemini 2.5 pro answer:
    #answer = """Based on the search results, the most effective interventions for reducing bat fatalities at wind turbines are: \"Prevent turbine blades from turning at low wind speeds ('feathering')\", \"Increase the wind speed at which turbines become operational ('cut-in speed')\", and \"Slow rotation of turbine blades at low wind speeds\". The first two are rated as 'Beneficial', and the third is rated as 'Likely to be beneficial'"""

    # Dodgy kimi k2 answer with incorrect stats:
    answer = "The most effective interventions for reducing bat fatalities at wind turbines are operational adjustments including preventing turbine blades from turning at low wind speeds (feathering) and increasing the wind speed at which turbines become operational (cut-in speed), both rated as beneficial based on strong evidence. Increasing cut-in speeds from the typical 3.5 m/s to 5.0-6.5 m/s significantly reduced bat mortality by 21-93% across multiple studies. Preventing turbine blades from turning at low wind speeds through feathering techniques also proved effective, reducing fatalities by 21-73% compared to conventional operations. Additionally, automatically reducing turbine blade rotation when bat activity is high has been identified as likely to be beneficial, with studies in Germany and the USA showing 25-92% reductions in fatalities. Slowing blade rotation at low wind speeds shows promise but has limited evidence, while ultrasonic deterrents have produced mixed and inconclusive results. These operational modifications balance conservation needs with energy production by targeting specific high-risk periods rather than continuous turbine shutdown, making them practical and cost-effective solutions for reducing bat mortality."

    evaluation = evaluate(question=question, answer=answer, docs=docs)
    print(evaluation)