import os
from dotenv import load_dotenv
import logging
from google import genai 
from google.genai import types, errors
from google.api_core import exceptions
from pydantic import BaseModel
from utils.action_parsing import ActionParsingContext, parse_action, get_parsed_action_as_str


load_dotenv()


class Evaluation(BaseModel):
    accuracy_score_reason : str
    action_ids_used_in_judgement : list[str]
    final_accuracy_score : str
    original_user_question: str
    potential_answer_to_question : str


def format_evaluation(accuracy_score_reason : str, action_ids_used_in_judgement : list[str], final_accuracy_score : str):
    """
    Format the evaluation results into a valid JSON object string.

    Args:
        accuracy_score_reason (str): A brief justification for the accuracy score assigned to the potential answer.
        action_ids_used_in_judgement (list[str]): List of action IDs used in the reasoning for the score assigned to the potential answer.
        final_accuracy_score (str): The final accuracy score assigned to the potential answer.

    Returns:
        dict: A dictionary representation of the formatted evaluation.
    """
    formatted_evaluation = {
        "accuracy_score_reason": accuracy_score_reason,
        "action_ids_used_in_judgement": action_ids_used_in_judgement,
        "final_accuracy_score": final_accuracy_score,
    }
    return formatted_evaluation


format_evaluation_function = {
    "name": "format_evaluation",
    "description": "Format the evaluation results (accuracy score reason, action ids used in the reasoning, and the accuracy score itself) into a valid JSON object string. Use this as the last step before presenting the answer to the user.",
    "parameters":{
        "type": "object",
        "properties": {
            "accuracy_score_reason":{
                "type": "string",
                "description": "A brief justification for the accuracy score assigned to the potential answer."
            },
            "action_ids_used_in_judgement":{
                "type": "array",
                "items": {"type":"string"},
                "description": "List of action IDs used in the reasoning for the score assigned to the potential answer."
            },
            "final_accuracy_score": {
                "type": "string",
                "description": "The final accuracy score assigned to the potential answer."
            }
        },
        "required": ["accuracy_score_reason","action_ids_used_in_judgement", "final_accuracy_score"]
    }
}


TOOL_MAPPING = {
    "format_evaluation": format_evaluation
}


def get_llm_evaluation(question, answer, docs):
    sys_prompt =  f"""{docs}

    Above is a document containing conservation actions, their effectiveness and key messages. This forms part of the Conservation Evidence living evidence database. Each action is prefixed with a numerical id.

    You are an expert evaluator of AI-generated answers to user questions about conservation.
    You must assess accuracy of the answers strictly according to the rules below.

    Your Task:
    1) Read the user question, the AI-generated answer, and the relevant database actions.
    2) Compare the answer against the database to see if it meets all accuracy criteria.
    3) Construct a reasoning for which score to assign based on the accuracy criteria and scoring rules.
    4) After you have formed the reasoning, assign a score from 0 to 5 following the scoring rules.
    5) Use the format_evaluation function to convert your evaluation into a valid JSON format.

    Accuracy Criteria:
    An answer is accurate if it satisfies all of the following:
    1) Factual Support
        Every statement is fully supported by the database.
        No hallucinations, invented facts, overgeneralizations, or misinterpretations.
        No unsupported conclusions drawn from otherwise correct facts.
    2) Citation Correctness
        Every statement is cited.
        Each cited action ID actually contains (or directly supports) the statement attributed to it.
        No fabricated, non-existent, or irrelevant citations.
    3) Misleading Framing or Presentation (includes statement ordering)
        The answer preserves meaning and context from the sources.
        No omission of critical context that would change understanding.
        No ambiguous language that could lead to incorrect interpretation.
        No mixing of unrelated facts in a way that alters meaning.
        Statements are well-ordered: 
            Important statements (those most directly responsive to the question and/or supported by stronger or more relevant evidence in the provided actions) should appear earlier. If the statement ordering materially shifts emphasis away from the most relevant/important evidence, treat this as misleading framing.
            Signals of importance include: direct relevance to the question, clarity/strength of evidence as reflected in the provided key messages/effectiveness, breadth of support across actions, and specificity to the asked context.
    4) Internal Consistency
        No contradictions within the answer.
        No contradictions with the database.

    Scoring Rules (0 to 5):
    5/5 — Perfect Accuracy: No factual errors; no misleading framing (including statement order); all citations correct  and all statements must be supported by citations.
    4/5 — Minor Issues Only: All facts correct. Only small, low-impact citation issues. Only low-importance / less essential facts missing citations. No misleading framing.
    3/5 — Factually Correct but Significant Citation Problems or Non-misleading Order Problems: All facts correct, but notable citation errors or missing citations for major facts, or notable prioritization issues in statement order that do not change meaning. No hallucinations.
    2/5 — Meaning-Changing Misleading Framing: No hallucinations, but misleading framing that changes meaning (including materially misleading statement order). Citation issues may also be present.
    1/5 — Serious Factual Errors: One or more factual inaccuracies, misleading framing, or citation problems. Cannot be trusted without major re-verification.
    0/5 — Completely Unreliable: Mostly hallucinated, fabricated citations, highly misleading or irrelevant.

    Severe penalty rules:
        Any factual error or hallucination → score cannot exceed 1.
        Any misleading framing that changes meaning → score cannot exceed 2.
        Citation errors alone may reduce score depending on impact (from minor deductions to major if key facts lack correct citations).


    Output Requirements:

    Your evaluation must include the following:
    1) A short judgement evaluating the potential answer according to the scoring rules, which gives reasons behind which score should be assigned to the answer. This should reference specific statements, citations, or order issues as needed.
    2) The final accuracy score you assign to the potential answer (i.e. "0" | "1" | "2" | "3" | "4" | "5")
    3) A list of the action ids you used in your judgement (unique list).
    Finally you MUST use the format_evaluation function. This will convert your reasoning for the accuracy score, your assigned accuracy score and the list of the action IDs you used in your evaluation into a valid JSON format.

    
    Here is the question: {question}
    Here is the potential answer: {answer}
    Evaluate this answer.
    """

    client = genai.Client()

    tools = types.Tool(function_declarations=[format_evaluation_function])

    contents = [
        types.Content(
            role="user", 
            parts=[
                types.Part(
                    text=sys_prompt
                )
            ]
        )
    ]

    response = client.models.generate_content(
        model="gemini-2.5-pro", 
        contents=contents,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=1024),
            tools = [tools]
        )
    )

    return response


def get_evaluation(question, answer, docs):
    result = {
        "question": question,
        "answer": answer,
        "evaluation" : None
    }
    llm_response = get_llm_evaluation(question=question, answer=answer, docs=docs)
    tool_call = llm_response.candidates[0].content.parts[0].function_call
    if tool_call is None:
        print(llm_response)
        raise ValueError("No tool call for formatting found in the response.")
    else:
        result["evaluation"] = TOOL_MAPPING[tool_call.name](**tool_call.args)
        return result


def get_oracle_actions(id_list, context : ActionParsingContext):
    doc_type = context.get_doc_type()
    if doc_type == "km":
        base_dir = "action_data/key_messages/km_all"
    elif doc_type == "bg_km":
        base_dir = "action_data/background_key_messages/bg_km_all"
    else:# invalid doc_type, return empty action list.
        return []

    parsed_actions = []
    for id in id_list:
        filename = f"action_{id}_clean.txt"
        filepath = os.path.join(base_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        parsed_actions.append(parse_action(action_string=content, context=context))

    return parsed_actions


def get_oracle_actions_as_str(id_list, context : ActionParsingContext):
    actions = get_oracle_actions(id_list=id_list, context=context)
    action_strings = []
    for action in actions:
        action_str = get_parsed_action_as_str(action=action)
        action_strings.append(action_str)
    full_str = "\n\n".join(action_strings)
    return full_str


def main():
    logging.basicConfig(level=logging.DEBUG, filename="logfiles/llm_judge_gemini_sdk_accuracy.log")
    logging.info("STARTING answer evaluation process.")

    context = ActionParsingContext(required_fields=["action_id", "action_title", "key_messages"])
    question = "What are the most effective interventions for reducing bat fatalities at wind turbines?"
    synopsis = "Bat Conservation"
    oracle_action_ids = ["970","1960","2939","971","968"]
    docs = get_oracle_actions_as_str(id_list=oracle_action_ids, context=context)

    # Claude opus 4 answer:
    #answer = "Based on scientific evidence, the most effective interventions for reducing bat fatalities at wind turbines are:\n\n**1. Prevent turbine blades from turning at low wind speeds ('feathering') - Rated as Beneficial**\n- Five out of six studies in the USA and Canada found that feathering turbine blades resulted in fewer bat fatalities compared to conventionally operated turbines\n- This involves keeping blades stationary or nearly stationary during low wind conditions\n\n**2. Increase the wind speed at which turbines become operational ('cut-in speed') - Rated as Beneficial**\n- Ten out of 12 studies in the USA and Canada found that increasing cut-in speeds resulted in fewer bat fatalities\n- This prevents turbines from operating during periods when bats are most active (typically low wind conditions)\n- Often implemented alongside feathering for maximum effectiveness\n\n**3. Slow rotation of turbine blades at low wind speeds - Rated as Likely to be beneficial**\n- A replicated, randomized, controlled study in Canada found reduced bat fatalities when turbine blades were slowed at low wind speeds\n- This represents a middle ground between full operation and complete feathering\n\n**4. Automatically reduce turbine blade rotation when bat activity is high - Rated as Likely to be beneficial**\n- Two replicated studies in Germany and the USA found fewer bat fatalities when blade rotation was automatically reduced based on predicted bat activity\n- Uses real-time or predictive models to adjust turbine operation\n\n**5. Deter bats from turbines using ultrasound - Rated as Unknown effectiveness**\n- Mixed results from studies in the USA, with effectiveness varying by species and conditions\n- Some studies showed 21-64% reductions in fatalities, while others showed minimal or inconsistent effects\n- More research needed to determine optimal deployment\n\nThe most reliable and proven interventions are operational curtailment strategies (feathering and increasing cut-in speeds), which have shown consistent benefits across multiple studies. These methods work because most bat fatalities occur during low wind speeds when bats are most active."
    
    # Gemini 2.5 pro answer (no citations):
    answer = """Based on the search results, the most effective interventions for reducing bat fatalities at wind turbines are: \"Prevent turbine blades from turning at low wind speeds ('feathering')\", \"Increase the wind speed at which turbines become operational ('cut-in speed')\", and \"Slow rotation of turbine blades at low wind speeds\". The first two are rated as 'Beneficial', and the third is rated as 'Likely to be beneficial'"""

    # Dodgy kimi k2 answer (incorrect stats):
    #answer = "The most effective interventions for reducing bat fatalities at wind turbines are operational adjustments including preventing turbine blades from turning at low wind speeds (feathering) and increasing the wind speed at which turbines become operational (cut-in speed), both rated as beneficial based on strong evidence. Increasing cut-in speeds from the typical 3.5 m/s to 5.0-6.5 m/s significantly reduced bat mortality by 21-93% across multiple studies. Preventing turbine blades from turning at low wind speeds through feathering techniques also proved effective, reducing fatalities by 21-73% compared to conventional operations. Additionally, automatically reducing turbine blade rotation when bat activity is high has been identified as likely to be beneficial, with studies in Germany and the USA showing 25-92% reductions in fatalities. Slowing blade rotation at low wind speeds shows promise but has limited evidence, while ultrasonic deterrents have produced mixed and inconclusive results. These operational modifications balance conservation needs with energy production by targeting specific high-risk periods rather than continuous turbine shutdown, making them practical and cost-effective solutions for reducing bat mortality."

    evaluation = get_evaluation(question=question, answer=answer, docs=docs)
    logging.info(f"Evaluation results: {evaluation}")
    print(evaluation)
    logging.info("FINISHED answer evaluation process.")


if __name__ == "__main__":
    main()