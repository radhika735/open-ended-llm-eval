import os
import json
from dotenv import load_dotenv
import logging
from openai import OpenAI
from pydantic import BaseModel


load_dotenv()


class Evaluation(BaseModel):
    accuracy_score_reason : str
    action_ids_used_in_judgement : list[str]
    final_accuracy_score : str


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
    "type": "function",
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
        "required": ["accuracy_score_reason","action_ids_used_in_judgement", "final_accuracy_score"],
        "additionalProperties": False
    },
    "strict": True
}

tools = [format_evaluation_function]

TOOL_MAPPING = {
    "format_evaluation": format_evaluation
}


response_schema = {
    "type": "json_schema",
    "json_schema": {
      "name": "Evaluation",
      "strict": True,
      "schema": {
        "type": "object",
        "properties": {
          "accuracy_score_reason": {
            "type": "string",
            "description": "A brief justification for the accuracy score assigned to the potential answer."
          },
          "action_ids_used_in_judgement": {
            "type": "array",
            "items": {"type":"string"},
            "description": "List of action IDs used in the reasoning for the score assigned to the potential answer."
          },
          "final_accuracy_score": {
            "type": "string",
            "description": "The final accuracy score assigned to the potential answer."
          }
        },
        "required": ["accuracy_score_reason", "action_ids_used_in_judgement", "final_accuracy_score"],
        "additionalProperties": False
      }
    }
}


def get_client():
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    client = OpenAI(
        api_key=f"{openrouter_api_key}",
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "Authorization": f"Bearer {openrouter_api_key}"
        }
    )
    return client


def get_llm_evaluation(question, answer, docs, model, provider):
    sys_prompt =  f"""{docs}

    Above is a document containing conservation actions, their effectiveness and key messages. This forms part of the Conservation Evidence living evidence database. Each action is prefixed with a numerical id.

    You are an expert evaluator of AI-generated answers to user questions about conservation.
    You must assess accuracy of the answers strictly according to the rules below.

    Your Task:
    1) Read the user question, the AI-generated answer, and the relevant database actions.
    2) Compare the answer against the database to see if it meets all accuracy criteria.
    3) Construct a reasoning for which score to assign based on the accuracy criteria and scoring rules.
    4) After you have formed the reasoning, assign a score from 0 to 5 following the scoring rules.
    

    Accuracy Criteria:
    An answer is accurate if it satisfies all of the following:
    1) Factual Support
        Every statement is fully supported by the database.
        No hallucinations, invented facts, overgeneralizations, or misinterpretations.
        No unsupported conclusions drawn from otherwise correct facts.
    2) Citation Correctness
        Every statement is cited
        Each cited action ID actually contains (or directly supports) the statement attributed to it.
        No fabricated, non-existent, or irrelevant citations.
        Repetition of citations is permitted.
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

    Scoring Rules (0 to 5):
    5/5 — Perfect Accuracy: No factual errors; no misleading framing (including statement order); all citations correct  and all statements must be supported by citations.
    4/5 — Minor Issues Only: All facts correct. Only small, low-impact citation issues. Only low-importance / less essential facts missing citations. No misleading framing.
    3/5 — Factually Correct but Significant Citation Problems or Non-misleading Order Problems: All facts correct, but notable citation errors or missing citations for major facts, or notable prioritization issues in statement order that do not change meaning. No hallucinations.
    2/5 — Meaning-Changing Misleading Framing: No hallucinations, but misleading framing that changes meaning (including materially misleading statement order). Citation issues may also be present.
    1/5 — Serious Factual Errors: One or more factual inaccuracies, misleading framing, or citation problems. Cannot be trusted without major re-verification.

    Severe penalty rules:
        Any factual error or hallucination → score cannot exceed 1.
        Any misleading framing that changes meaning → score cannot exceed 2.
        Citation errors alone may reduce score depending on impact (from minor deductions to major if key facts lack correct citations).
    Additional rules:
        Repeated citations should NOT be penalised.

    Output Requirements:

    Your evaluation must include the following:
    1) A short judgement evaluating the potential answer according to the scoring rules, which gives reasons behind which score should be assigned to the answer. This should reference specific statements, citations, or order issues as needed.
    2) A list of the action ids you used in your judgement (unique list).
    3) The final accuracy score you assign to the potential answer (i.e. "0" | "1" | "2" | "3" | "4" | "5")
    Format your answer as a valid JSON object with the following schema:
    {{
        "accuracy_score_reasoning": str,
        "action_ids_used_in_judgement": list[str],
        "final_accuracy_score": "0|1|2|3|4|5"
    }}
    """

    user_prompt = f"""
    Here is a question: {question}
    Here is a potential answer: {answer}
    Evaluate this answer.
    """

    client = get_client()

    messages = [
        {
            "role": "system",
            "content": sys_prompt
        },
        {
            "role": "user",
            "content": user_prompt,
        }
    ]

    if provider is not None:
        response = client.chat.completions.create(
            model = model,
            messages = messages,
            reasoning_effort = "low",
            response_format = response_schema,
            extra_body={
                "provider" : {
                    #"require_parameters" : True,
                    "order": [f"{provider}"],
                    "allow_fallbacks": False
                }
            }
        )
    else:
        response = client.chat.completions.create(
            model = model,
            messages = messages,
            reasoning_effort = "low",
            extra_body={
                "require_parameters": True
            }
        )

    return response.choices[0].message.content


def get_evaluation(question, answer, docs, model="google/gemini-2.5-pro", provider=None):
    result = {
        "question": question,
        "answer": answer,
        "evaluation" : None
    }
    llm_json = get_llm_evaluation(question=question, answer=answer, docs=docs, model=model, provider=provider)
    result["evaluation"] = json.loads(llm_json)
    return result


def get_oracle_actions(oracle_ids):
    # doc_type can be "km" or "bg_km", with "km" for key messages and "bg_km" for background key messages
    """
    Get parsed actions from the data directory.
    
    Returns:
        list: List of parsed action dictionaries
    """
    oracle_actions = []
    
    data_dir = f"action_data/key_messages/km_all"

    for id in oracle_ids:
        filename = os.path.join(data_dir, f"action_{id}_clean.txt")
        with open(filename, "r", encoding="utf-8") as action_file:
            file_contents = action_file.read()
        oracle_actions.append(file_contents)

    return oracle_actions


def get_oracle_actions_str(oracle_ids):
    doc_strs = get_oracle_actions(oracle_ids=oracle_ids)
    full_str = "\n\n".join(doc_strs)
    return full_str


def parse_model_name(model):
    model_split = model.split("/")
    model_name = model_split[-1]
    cleaned_name = ""
    for char in model_name:
        if char.isalnum():
            cleaned_name += char
        else:
            cleaned_name += "-"
    return cleaned_name


def parse_provider_name(provider):
    if provider is not None:
        provider_split = provider.split("/")
        provider_name = provider_split[0]
        return provider_name
    else:
        return ""


def main():
    logging.basicConfig(level=logging.DEBUG, filename="logfiles/llm_judge_openai_sdk_accuracy.log")
    # logging.info("STARTING answer evaluation process.")
    # question = "What are the most effective interventions for reducing bat fatalities at wind turbines?"
    # synopsis = "Bat Conservation"
    # oracle_action_ids = ["970","1960","2939","971","968"]
    # docs = get_oracle_actions_str(synopsis=synopsis, oracle_ids=oracle_action_ids)

    # # Claude opus 4 answer:
    # #answer = "Based on scientific evidence, the most effective interventions for reducing bat fatalities at wind turbines are:\n\n**1. Prevent turbine blades from turning at low wind speeds ('feathering') - Rated as Beneficial**\n- Five out of six studies in the USA and Canada found that feathering turbine blades resulted in fewer bat fatalities compared to conventionally operated turbines\n- This involves keeping blades stationary or nearly stationary during low wind conditions\n\n**2. Increase the wind speed at which turbines become operational ('cut-in speed') - Rated as Beneficial**\n- Ten out of 12 studies in the USA and Canada found that increasing cut-in speeds resulted in fewer bat fatalities\n- This prevents turbines from operating during periods when bats are most active (typically low wind conditions)\n- Often implemented alongside feathering for maximum effectiveness\n\n**3. Slow rotation of turbine blades at low wind speeds - Rated as Likely to be beneficial**\n- A replicated, randomized, controlled study in Canada found reduced bat fatalities when turbine blades were slowed at low wind speeds\n- This represents a middle ground between full operation and complete feathering\n\n**4. Automatically reduce turbine blade rotation when bat activity is high - Rated as Likely to be beneficial**\n- Two replicated studies in Germany and the USA found fewer bat fatalities when blade rotation was automatically reduced based on predicted bat activity\n- Uses real-time or predictive models to adjust turbine operation\n\n**5. Deter bats from turbines using ultrasound - Rated as Unknown effectiveness**\n- Mixed results from studies in the USA, with effectiveness varying by species and conditions\n- Some studies showed 21-64% reductions in fatalities, while others showed minimal or inconsistent effects\n- More research needed to determine optimal deployment\n\nThe most reliable and proven interventions are operational curtailment strategies (feathering and increasing cut-in speeds), which have shown consistent benefits across multiple studies. These methods work because most bat fatalities occur during low wind speeds when bats are most active."
    
    # # Gemini 2.5 pro answer (no citations):
    # #answer = """Based on the search results, the most effective interventions for reducing bat fatalities at wind turbines are: \"Prevent turbine blades from turning at low wind speeds ('feathering')\", \"Increase the wind speed at which turbines become operational ('cut-in speed')\", and \"Slow rotation of turbine blades at low wind speeds\". The first two are rated as 'Beneficial', and the third is rated as 'Likely to be beneficial'"""

    # # Dodgy kimi k2 answer (incorrect stats):
    # answer = "The most effective interventions for reducing bat fatalities at wind turbines are operational adjustments including preventing turbine blades from turning at low wind speeds (feathering) and increasing the wind speed at which turbines become operational (cut-in speed), both rated as beneficial based on strong evidence. Increasing cut-in speeds from the typical 3.5 m/s to 5.0-6.5 m/s significantly reduced bat mortality by 21-93% across multiple studies. Preventing turbine blades from turning at low wind speeds through feathering techniques also proved effective, reducing fatalities by 21-73% compared to conventional operations. Additionally, automatically reducing turbine blade rotation when bat activity is high has been identified as likely to be beneficial, with studies in Germany and the USA showing 25-92% reductions in fatalities. Slowing blade rotation at low wind speeds shows promise but has limited evidence, while ultrasonic deterrents have produced mixed and inconclusive results. These operational modifications balance conservation needs with energy production by targeting specific high-risk periods rather than continuous turbine shutdown, making them practical and cost-effective solutions for reducing bat mortality."

    # evaluation = get_evaluation(question=question, answer=answer, docs=docs)
    # logging.info(f"Evaluation results: {evaluation}")
    # print(evaluation)
    # logging.info("FINISHED answer evaluation process.")


    # question = "What actions can be taken to mitigate the environmental pollution caused by waste from salmon farms?"
    # answer = "A quick summary: Options for mitigating pollution from salmon farm waste include integrated aquaculture systems, regulatory bans on pollutant disposal, and pollutant removal technologies, though evidence for effectiveness varies.\n\nOne notable method is the use of integrated aquaculture systems, where species like scallops or mussels are cultivated near salmon farms to absorb or utilize farm waste. For example, studies have shown greater growth of scallops and absorption of salmon farm waste by blue mussels in some locations, as well as enhanced algal growth near fish farms, though at least one study found no evidence of shellfish feeding on fish waste, so results are inconsistent (932). Regulatory bans on the marine burial of persistent pollutants, such as those encompassing salmon farm-related waste, have been proposed, but there is no direct evidence assessing their effectiveness on environmental improvement specific to salmon farms (3595). Another general approach—removing pollutants from waste gases before environmental release—was assessed in a different context (bogs) and produced mixed results for plant diversity, yet its application to salmon farming waste is not directly evidenced (1789). While integrated aquaculture systems seem a promising method, regulatory solutions and gas waste removal also offer potential, although the best outcomes may arise from other, yet unassessed, interventions. Interestingly, despite awaiting full effectiveness assessments, integrated aquaculture could be most effective, yet the lack of direct evaluations makes both bans and modern filtration methods equally plausible as optimal actions depending on context (932, 3595, 1789)."
    # oracle_ids = ["932","3595","1789"]

    # question = "What are the most effective ways to increase soil organic carbon on loamy soils?"
    # answer = "Reducing grazing intensity, amending soil with formulated chemical compounds, and changing tillage practices are effective ways to increase soil organic carbon on loamy soils.\n\nA likely beneficial approach is to reduce grazing intensity: replicated studies show that excluding or reducing livestock grazing increases soil carbon and nitrogen, helps compacted soils recover, and reduces sediment erosion and nutrient loss on loamy soils (action_id: 901). Amending soil with chemical compounds such as nitrogen or phosphorus fertilizers also often raises soil organic matter or carbon levels, with evidence from several replicated and controlled studies, particularly when combined with organic material inputs; however, manure tends to be even more effective than chemical compounds alone (action_id: 909). Changing tillage practices\u2014especially shifting to no-tillage or reduced tillage\u2014consistently increases soil organic carbon compared to conventional ploughing, as demonstrated in multiple large and well-designed studies. However, this method comes with trade-offs such as potential increases in soil compaction or mixed effects on drought resistance, nutrient retention, and yields (action_id: 906). Together, evidence supports that grazing reduction and careful management of nutrient amendments and tillage are key strategies for increasing soil organic carbon in loamy soils. \n\nReferences: action_id: 901, action_id: 909, action_id: 906."
    # oracle_ids = ["906","857","902","907","911"]
    
    # docs = get_oracle_actions_str(oracle_ids=oracle_ids)
    # evaluation = get_evaluation(question=question, answer=answer, docs=docs)
    # print(evaluation)
    # print("end")


    ## MINI TEST ON HUMAN AGREEMENT WITH LLM JUDGE:

    oracle_id_table = {
        "What are the most beneficial actions for reducing human-wildlife conflict with bears?": ["2330","2336","2346","2347","2385"],
        "What actions can be taken to mitigate the environmental pollution caused by waste from salmon farms?": ["1027","932","934","943"],
        "What are the most effective ways to increase soil organic carbon on loamy soils?": ["906","857","902","907","911"]
    }

    with open("evaluation_data/mini_testing_human_agreement/final_answers_to_test.json", "r", encoding="utf-8") as f:
        final_answers = json.load(f)

    model = "qwen/qwen3-235b-a22b-thinking-2507"
    provider = "deepinfra/fp8"

    evaluations = []

    for qu_ans_dict in final_answers:
        question = qu_ans_dict["query"]
        answer = qu_ans_dict["answer"]
        action_ids = qu_ans_dict["action_ids"]
        oracle_ids = oracle_id_table.get(question, [])
        all_ids = list(set(action_ids) | set(oracle_ids))

        docs = get_oracle_actions_str(oracle_ids=all_ids)
        evaluation = get_evaluation(question=question, answer=answer, docs=docs, model=model, provider=provider)
        evaluation["action_ids_given_to_judge"] = all_ids
        evaluations.append(evaluation)
    
    cleaned_model_name = parse_model_name(model=model)
    cleaned_provider_name = parse_provider_name(provider=provider)
    with open(f"evaluation_data/mini_testing_human_agreement/evaluation_results_{cleaned_provider_name}_{cleaned_model_name}.json", "w", encoding="utf-8") as f:
        json.dump(evaluations, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()