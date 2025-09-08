import os
import json
from dotenv import load_dotenv
import logging
from openai import OpenAI
from utils.action_retrieval import ActionRetrievalContext, parse_action, get_parsed_action_as_str


load_dotenv()


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
    "function": {
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
}

tools = [format_evaluation_function]

TOOL_MAPPING = {
    "format_evaluation": format_evaluation
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
    
    Finally, you MUST use the format_evaluation tool. This will convert your generated judgement, your assigned accuracy score and the list of the action IDs you used in your judgement into a valid JSON format.
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
            tools = tools,
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
            tools = tools,
            extra_body={
                "require_parameters": True
            }
        )

    return response


def execute_tool_call(tool_call):
    try:
        print(tool_call)
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        logging.info(f"Executing tool: {name} with args: {args}")
        print(args)
        print(type(args))
        # Execute the tool function
        tool_result = TOOL_MAPPING[name](**args)
        tool_success = True
        return tool_result, tool_success
    except (AttributeError, KeyError, TypeError) as e:
        logging.error(f"Error occurred while executing tool: {e}")
        tool_result = "NONE - Erroneous tool call"
        tool_success = False
        return tool_result, tool_success
        

def get_evaluation(question, answer, docs, model="google/gemini-2.5-pro", provider=None):
    result = {
        "question": question,
        "answer": answer,
        "evaluation" : None
    }
    llm_response = get_llm_evaluation(question=question, answer=answer, docs=docs, model=model, provider=provider)
    if llm_response.choices[0].message.tool_calls:
        logging.info("Requested tool call(s).")
        if len(llm_response.choices[0].message.tool_calls) == 1:
            tool_call = llm_response.choices[0].message.tool_calls[0]
            tool_result, tool_success = execute_tool_call(tool_call=tool_call)
            if tool_success:
                evaluation = tool_result
            else:
                evaluation = llm_response.choices[0].message.content
        else:
            logging.warning("Requested more than one tool calls.")
            evaluation = llm_response.choices[0].message.content
    else:
        logging.warning("Did not request a tool call")
        evaluation = llm_response.choices[0].message.content

    result["evaluation"] = evaluation
    return result


def get_oracle_actions(id_list, context : ActionRetrievalContext):
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


def get_oracle_actions_as_str(id_list, context : ActionRetrievalContext):
    actions = get_oracle_actions(id_list=id_list, context=context)
    action_strings = []
    for action in actions:
        action_str = get_parsed_action_as_str(action=action)
        action_strings.append(action_str)
    full_str = "\n\n".join(action_strings)
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
    logging.basicConfig(level=logging.INFO, filename="logfiles/llm_judge_openai_sdk_accuracy_tools.log", format='%(asctime)s - %(levelname)s - %(message)s')

    ## MINI TEST ON HUMAN AGREEMENT WITH LLM JUDGE:
    logging.info("STARTING evaluation generation.")
    
    context = ActionRetrievalContext(required_fields=["action_id", "action_title", "key_messages"])

    oracle_id_table = {
        "What are the most beneficial actions for reducing human-wildlife conflict with bears?": ["2330","2336","2346","2347","2385"],
        "What actions can be taken to mitigate the environmental pollution caused by waste from salmon farms?": ["1027","932","934","943"],
        "What are the most effective ways to increase soil organic carbon on loamy soils?": ["906","857","902","907","911"]
    }

    print("loading qus and answers")
    with open("evaluation_data/mini_testing_human_agreement/final_answers_to_test.json", "r", encoding="utf-8") as f:
        final_answers = json.load(f)

    model = "openai/gpt-5"
    provider = None

    evaluations = []

    for i,qu_ans_dict in enumerate(final_answers):
        print("Iteration",i)
        question = qu_ans_dict["query"]
        answer = qu_ans_dict["answer"]
        action_ids = qu_ans_dict["action_ids"]
        oracle_ids = oracle_id_table.get(question, [])
        all_ids = list(set(action_ids) | set(oracle_ids))

        docs = get_oracle_actions_as_str(id_list=all_ids, context=context)
        evaluation = get_evaluation(question=question, answer=answer, docs=docs, model=model, provider=provider)
        evaluation["action_ids_given_to_judge"] = all_ids
        evaluations.append(evaluation)
        print(evaluation)
        print("Added evaluation for iteration",i)
    
    cleaned_model_name = parse_model_name(model=model)
    cleaned_provider_name = parse_provider_name(provider=provider)
    with open(f"evaluation_data/mini_testing_human_agreement/evaluation_results_{cleaned_provider_name}_{cleaned_model_name}.json", "w", encoding="utf-8") as f:
        json.dump(evaluations, f, ensure_ascii=False, indent=2)

    logging.info("ENDING evaluation generation.")




if __name__ == "__main__":
    main()