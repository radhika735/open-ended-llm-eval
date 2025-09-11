# some parts copied heavily from "Evaluation of RAG Metrics for Question Answering in the Telecom Domain" paper
import os
import json
import logging
import re
from dotenv import load_dotenv
from openai import OpenAI
from google import genai
from google.genai import types
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pydantic import BaseModel, Field

from utils import action_parsing

load_dotenv()



### API CALLING.

def get_client(api="openrouter"):
    """
    Get an OpenAI client configured for Openrouter or a GenAI client configured for the Gemini API.

    """
    if api.lower() == "openrouter":# default to openai sdk, ignoring value passed to sdk parameter
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouter")
        base_url = "https://openrouter.ai/api/v1"
        return OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    elif api.lower() == "gemini":
        return genai.Client()


def call_llm(messages, api="openrouter", model="google/gemini-2.5-pro", max_tokens=15000, reasoning_effort=None, tools=None, response_format=None):
    if api.lower() == "openrouter":
        client = get_client(api="openrouter")
        request_params = {
            "model":model,
            "messages": messages,
            "max_tokens": max_tokens,
            "extra_body": {
                "require_parameters": True,
                "reasoning":{
                    "enabled": True,
                    "effort": reasoning_effort if reasoning_effort is not None else "high"
                },
            }
        }

        if tools is not None:
            request_params.update({"tools": tools})
        elif response_format is not None:
            request_params.update({"response_format": response_format})
        response = client.chat.completions.create(**request_params)
    
    elif api.lower() == "gemini":
        client = get_client(api="gemini")
        thinking_tokens = 1024 if reasoning_effort == "low" else 2048 if reasoning_effort == "medium" else 4096
        request_params = {
            "model": model,
            "contents": messages,
        }
        if tools is not None:
            request_params.update(
                {
                    "config": types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=thinking_tokens),
                        tools=tools,
                    )
                }
            )
        elif response_format is not None:
            request_params.update(
                {
                    "config": types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=thinking_tokens),
                        response_mime_type= "application/json",
                        response_schema = response_format
                    )
                }
            )
        response = client.models.generate_content(**request_params)
            
    
    return response



### STATEMENT EXTRACTION.

def get_statements(question, answer):
    ## RAGAS statement splitting prompt:
    prompt = f"""
        Given a question and answer, create one or more statements from each sentence in the given answer. Output the statements in the following format strictly. [Statement n]: ... where the n is the statement number.\nquestion: {question}\nanswer: {answer}\nStatements:\n
    """.strip()

    ## Own prompt:
    prompt = f"""
Given a question and answer, create one or more statements from each sentence in the given answer. Each statement should be fully understandable by itself, which means it needs to be self-contained. This means there should be no pronouns in the statements. Output the statements in the following format strictly. [Statement n]: ... where the n is the statement number.\nquestion: {question}\nanswer: {answer}\nStatements:\n
    """.strip()

    messages=[
        {"role": "user", "content": prompt}
    ]

    response = call_llm(messages=messages).choices[0].message.content

    # Splitting statements
    response = re.sub(
        r"^.*?[0-9]+\]*[:\.]\s(.*?)", r"\1", response, flags=re.MULTILINE
    ).strip()
    statements = sent_tokenize(response)
    return statements


class CitedStatement(BaseModel):
    statement : str
    citations : list[str]


def get_citations_from_statements(summary, statements):
    prompt = f"""
Given a summary of information and a list of statements extracted from this summary, you must extract the document IDs cited in the summary for each statement. There may be zero or more cited IDs per statement.
Output the statements with their extracted citations as a list of JSON objects.

Summary: {summary}
Statements: {statements}
    """.strip()

    messages = [
        {"role": "user", "content": prompt}
    ]

    cited_statements_response_schema = {
        "type": "array",
        "items" : CitedStatement.model_json_schema()
    }
    cited_statements_response_format = {
        "type": "json_schema",
        "json_schema": {
            "name" : "ListOfCitedStatements",
            "strict" : True, 
            "schema": cited_statements_response_schema
        }
    }

    llm_response = call_llm(messages=messages, response_format=cited_statements_response_format)
    response_content = llm_response.choices[0].message.content
    cited_statements = json.loads(response_content)
    return cited_statements



### METRIC EVALUATION. Assessing metrics: faithfulness, answer relevance, citation correctness.


## JSON Schema wrapping individual judgement:
class StatementVerdict(BaseModel):
    statement: str = Field(description="A given statement.")
    reasoning: str = Field(description="The reasoning behind the verdict.")
    verdict: str = Field(description="The verdict for the statement (i.e., 'Yes' or 'No').")

## Existing RAGAS faithfulness metric:
def faithfulness(question, summary, docs, statements=None):
    if statements is None:
        statements = get_statements(question=question, answer=summary)
    statements_str = "\n".join([f"[Statement {i+1}: {s}]" for i, s in enumerate(statements)])

    ## True RAGAS prompt:
#     prompt = f"""
# Consider the given context and following statements, then determine whether they are supported by the information present in the context. Provide a brief explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order. Output the in-order list of statements, a list of their verdicts in-order, and a list of the reasonings for the verdicts in-order as a JSON object.\nContext: {docs}\nStatements: {statements_str}\nResponse:\n
#     """.strip()

    ## Own, RAGAS-inspired but modified prompt:
    prompt = f"""
Consider the given context and following statements, then determine whether they are supported by the information present in the context. ALL parts of the statement must be FULLY supported for it to be assigned a 'Yes' verdict. Provide a thorough and rigorous explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order. Output the statements, reasonings and verdicts as a valid list of JSON objects.\nContext: {docs}\nStatements: {statements_str}\nResponse:\n
    """.strip()

    messages = [
        {"role": "user", "content": prompt}
    ]

    ## Response format using JSON Schema wrapping individual judgements:
    verdicts_response_format = {
        "type" : "json_schema",
        "json_schema" : {
            "name" : "Statements, Verdicts and Reasonings",
            "strict" : True,
            "schema" : {
                "type": "array",
                "items" : StatementVerdict.model_json_schema()
            }
        }
    }
    raw_response = call_llm(messages=messages, response_format=verdicts_response_format)

    # Usage details for logging purposes:
    usage_details = raw_response.usage
    print(f"Token usage: completion_tokens={usage_details.completion_tokens} (reasoning_tokens={usage_details.completion_tokens_details.reasoning_tokens}), prompt_tokens={usage_details.prompt_tokens}, total_tokens={usage_details.total_tokens}, cached_tokens={usage_details.prompt_tokens_details.cached_tokens}\n")
    
    response = raw_response.choices[0].message.content
    response = json.loads(response)

    ## Extracting output when using JSON Schema wrapping individual judgements:
    judgements = []
    for obj in response:
        judgements.append({
            "statement": obj["statement"],
            "reasoning": obj["reasoning"],
            "verdict": True if "yes" in obj["verdict"].lower() else False
        })
    verdicts = [obj["verdict"] for obj in judgements]

    num_supported_statements = sum(1 for v in verdicts if v)
    total_statements = len(verdicts)
    score = (num_supported_statements / total_statements) if total_statements > 0 else 0

    return {
        "score":score,
        "judgements":judgements
    }


## TO DELETE AND USE THE ONE IN UTILS
def embed(text, model_name="nomic-ai/nomic-embed-text-v1.5"):
    model = SentenceTransformer(model_name, trust_remote_code=True)
    embeddings = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
    return embeddings


## Not using this metric for now, since it seems to unfairly penalise summaries.: 
#   Since we are generating summaries of evidence rather than answers to the question directly, 
#   we feel that the synthetic questions generated are by default further away from the original question than in a typical QA setting,
#   leading to summaries being unfairly penalised in terms of relevance / completely answering the question.
def answer_relevance(query, answer, n=10):
    prompt = f"""
Generate {n} potential questions for the given answer. Output the questions in the following format strictly. [Question n]: ... where the n is the question number.\nanswer: {answer}\nQuestions:\n
    """.strip()
    messages = [
        {"role": "user", "content": prompt}
    ]
    response = call_llm(messages=messages).choices[0].message.content
    response = response.strip()
    parsed_response = re.sub(
        r"^.*?[0-9]+\]*[:\.]\s(.*?)", r"\1", response, flags=re.MULTILINE
    ).strip()
    gen_questions = sent_tokenize(parsed_response)

    rephrased_query = "Provide a summary of evidence relevant to the following question: " + query
    original_qu_embedding = embed(rephrased_query)
    gen_qus_embeddings = embed(gen_questions)

    answer_relevance_scores = []
    for gen_embedding in gen_qus_embeddings:
        sim_score = cosine_similarity(original_qu_embedding.reshape(1, -1), gen_embedding.reshape(1, -1))
        answer_relevance_scores.append(sim_score)

    avg_score = np.mean(answer_relevance_scores) if answer_relevance_scores else 0
    avg_score = float(avg_score)
    return {
        "score": avg_score,
        "questions": gen_questions
    }


## JSON schema wrapping individual judgement:
class CitedStatementVerdict(BaseModel):
    statement: str = Field(description="A given statement.")
    citations: list[str] = Field(description="The list of citations for the statement (list).")
    reasoning: str = Field(description="The reasoning behind the verdict.")
    verdict: str = Field(description="The verdict for the statement (i.e., 'Yes' or 'No').")


## New metric: citation correctness within the summary.
#   Judges the "accuracy" of the citations themeselves for each statement, i.e. whether the citations given for each statement actually support the statement.
#   Statements with no citations are consequently judges as not supported by default, meaning unsupported statements are automatically penalised.
#   A high score is indicative of most facts being cited and those cited documents actually supporting the facts.
def citation_correctness(question, summary, docs, statements=None):
    if statements is None:
        statements = get_statements(question=question, answer=summary)
    cited_statements = get_citations_from_statements(summary=summary, statements=statements)
    statements_joined_citations = [{"statement":obj["statement"], "citations": ", ".join(obj["citations"])} for obj in cited_statements]
    cited_statements_str = "\n".join([f"[Statement {i+1}: {obj["statement"]}\nCitations for statement {i+1}: {obj["citations"]}]" for i, obj in enumerate(statements_joined_citations)])

    ## Least alterations to RAGAS faithfulness prompt:
    # prompt = f"""
    #     Consider the given context and following statements, then determine whether they are FULLY supported by the information present in the context CITED for the corresponding statement. If a statement has no citations then it is not supported by default. Provide a thorough explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order. Output the in-order list of statements, a list of the citations for each statement, a list of the statement verdicts in-order, and a list of the reasonings for the verdicts in-order as a JSON object.\nContext: {docs}\nStatements with their citations: {cited_statements_str}\nResponse:\n
    # """.strip()
#     prompt = f"""
# Consider the given context and following statements, then determine whether they are supported by the information present in the context CITED for the corresponding statement. ALL parts of the statement must be FULLY supported for it to be assigned a 'Yes' verdict. If a statement has no citations then it is not supported by default. Provide a thorough and rigorous explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order. Output the statements, the list of citations for each statement, reasonings and verdicts as a valid list of JSON objects.\nContext: {docs}\nStatements: {cited_statements_str}\nResponse:\n
#     """.strip()

    ## Based on own, RAGAS-inspired but modified, faithfulness prompt:
    prompt = f"""
Consider the given context and following statements, then determine whether they are supported by the information present in the context CITED for the corresponding statement. ALL parts of the statement must be FULLY supported for it to be assigned a 'Yes' verdict. If a statement has no citations then it is not supported by default. Provide a thorough and rigorous explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order. Output the statements, the list of citations for each statement, reasonings and verdicts as a valid list of JSON objects.\nContext: {docs}\nStatements: {cited_statements_str}\nResponse:\n
    """.strip()
    
    messages = [
        {"role": "user", "content": prompt}
    ]

    ## Response format using JSON schema wrapping individual judgements:
    verdicts_response_format = {
        "type" : "json_schema",
        "json_schema" : {
            "name" : "Statements, Citations, Verdicts and Reasonings",
            "strict" : True,
            "schema": {
                "type": "array",
                "items" : CitedStatementVerdict.model_json_schema()
            }
        }
    }

    raw_response = call_llm(messages=messages, response_format=verdicts_response_format)

    # Usage details for logging purposes:
    usage_details = raw_response.usage
    print(f"Token usage: completion_tokens={usage_details.completion_tokens} (reasoning_tokens={usage_details.completion_tokens_details.reasoning_tokens}), prompt_tokens={usage_details.prompt_tokens}, total_tokens={usage_details.total_tokens}, cached_tokens={usage_details.prompt_tokens_details.cached_tokens}\n")

    response = raw_response.choices[0].message.content
    response = json.loads(response)

    ## Extracting output when using JSON schema wrapping individual judgements:
    # response_statements = []
    # response_citations = []
    # verdicts = []
    # reasonings = []
    # for obj in response:
    #     response_statements.append(obj["statement"])
    #     response_citations.append(obj["citations"])
    #     verdicts.append(True if "yes" in obj["verdict"].lower() else False)
    #     reasonings.append(obj["reasoning"])
    judgements = []
    for obj in response:
        judgements.append({
            "statement": obj["statement"],
            "citations": obj["citations"],
            "reasoning": obj["reasoning"],
            "verdict": True if "yes" in obj["verdict"].lower() else False
        })
    verdicts = [obj["verdict"] for obj in judgements]
    

    num_supported_statements = sum(1 for v in verdicts if v)
    total_statements = len(verdicts)
    score = (num_supported_statements / total_statements) if total_statements > 0 else 0

    return {
        "score":score,
        "judgements":judgements
    }


## New metric: relevance of the summary to the question.
#   Developed this metric due to not using answer_relevance metric.
#   This and a completeness metric would be a substitute to answer_relevance which takes into account both relevance of answer and completeness of it.
#   The metric works by splitting the summary into statements, and giving a verdict on each one as to whether it is relevant to answering the question (similar to RAGAS faithfulness procedure).
#   A high score is indicative of many statements in the summary being relevant to answering the question.
def relevance(question, summary, statements=None):
    if statements is None:
        statements = get_statements(question=question, answer=summary)
    statements_str = "\n".join([f"[Statement {i+1}: {s}]" for i, s in enumerate(statements)])
    prompt = f"""
Given a question and a list of statements, determine whether each statement is relevant to answering the question. Provide a thorough and rigorous explanation for each statement before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order. Output the statements, the list of citations for each statement, reasonings and verdicts as a valid list of JSON objects.\nQuestion: {question}\nStatements: {statements_str}\nResponse:\n
    """.strip()

    messages = [
        {"role": "user", "content": prompt}
    ]

    verdicts_response_format = {
        "type" : "json_schema",
        "json_schema" : {
            "name" : "Statements, Verdicts and Reasonings",
            "strict" : True,
            "schema" : {
                "type": "array",
                "items" : StatementVerdict.model_json_schema()
            }
        }
    }
    raw_response = call_llm(messages=messages, response_format=verdicts_response_format)

    # Usage details for logging purposes:
    usage_details = raw_response.usage
    print(f"Token usage: completion_tokens={usage_details.completion_tokens} (reasoning_tokens={usage_details.completion_tokens_details.reasoning_tokens}), prompt_tokens={usage_details.prompt_tokens}, total_tokens={usage_details.total_tokens}, cached_tokens={usage_details.prompt_tokens_details.cached_tokens}\n")
    
    response = raw_response.choices[0].message.content
    response = json.loads(response)

    ## Extracting output when using JSON Schema wrapping individual judgements:
    # response_statements = []
    # verdicts = []
    # reasonings = []
    # for obj in response:
    #     response_statements.append(obj["statement"])
    #     verdicts.append(True if "yes" in obj["verdict"].lower() else False)
    #     reasonings.append(obj["reasoning"])
    judgements = []
    for obj in response:
        judgements.append({
            "statement": obj["statement"],
            "citations": obj["citations"],
            "reasoning": obj["reasoning"],
            "verdict": True if "yes" in obj["verdict"].lower() else False
        })
    verdicts = [obj["verdict"] for obj in judgements]

    num_relevant_statements = sum(1 for v in verdicts if v)
    total_statements = len(verdicts)
    score = (num_relevant_statements / total_statements) if total_statements > 0 else 0

    return {
        "score":score,
        "judgements":judgements
    }


## Here until 'ACTION DOC PARSING' implements a completeness metric, implementation unfinished.
#   This metric works by splitting up the ground truth context for the question (i.e. the key messages of the action documents) into sentences, 
#       and giving a verdict on each one as to whether it is relevant to answering the question (similar to RAGAS faithfulness procedure).
#   The sentences in the context given a positive verdict are considered the 'relevant sentences' required for completeness of the summary.
#   Then the completeness score is the proportion of the relevant sentences that are used / mentioned / covered in the summary.
#   The ground truth context is taken as the all_relevant_action_ids field produced during question generation (not necessarily comprehensive but ).

# class SentenceVerdict(BaseModel):
#     sentence: str = Field(description="A given sentence from the context.")
#     reasoning: str = Field(description="The reasoning behind the verdict.")
#     verdict: str = Field(description="The verdict for the sentence (i.e., 'Yes' or 'No').")
def format_sentence_verdicts(sentences, reasonings, verdicts):
    cleaned_verdicts = [True if "yes" in v.lower() else False for v in verdicts]
    num_relevant_sentences = sum(1 for v in cleaned_verdicts if v)
    total_sentences = len(cleaned_verdicts)
    score = (num_relevant_sentences / total_sentences) if total_sentences > 0 else 0
    formatted = {
        "score": score,
        "sentences": sentences,
        "reasonings": reasonings,
        "verdicts": cleaned_verdicts
    }

format_sentence_verdicts_function = {
    "type":"function",
    "function": {
        "name": "format_sentence_verdicts",
        "description": "Format the list of sentences, reasonings and verdicts into a valid JSON objects. Use this as the last step before presenting your results to the user.",
        "parameters": {
            "sentences":{
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of the sentences given in the context."
            },
            "reasonings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "A list of the reasonings corresponding to each sentence."
            },
            "verdicts": {
                "type": "array",
                "items": {"type": "boolean"},
                "description": "A list of the verdicts corresponding to each sentence."
            },
            "required": ["sentences", "reasonings", "verdicts"],
            "additionalProperties": False
        },
        "strict": True
    }
}

completeness_tools = [format_sentence_verdicts_function]

COMPLETENESS_TOOL_MAPPING = {
    "format_sentence_verdicts": format_sentence_verdicts
}

def execute_completeness_tool_call(tool_call):
    try:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        logging.info(f"Executing tool: {name} with args: {args}")
        # Execute the tool function
        tool_result = COMPLETENESS_TOOL_MAPPING[name](**args)
        tool_success = True
        return tool_result, tool_success
    except (AttributeError, KeyError, TypeError) as e:
        logging.error(f"Error occurred while executing tool: {e}")
        tool_result = "NONE - Erroneous tool call"
        tool_success = False
        return tool_result, tool_success


def completeness(question, summary, docs):
    doc = docs[0]
    context = doc["key_messages"]

    ## Prompt for structured output
#     prompt = f"""
# Consider the given context and question, then for each sentence in the context, determine whether it is relevant to answering the question. Provide a brief explanation of why each sentence is relevant or not before arriving at the verdict (Yes/No). Provide a final verdict for each sentence in order. \nContext: {context}\nQuestion: {question}\n\nOutput the in-order sentences, reasonings and verdicts as a valid list of JSON objects.\nResponse:\n
#     """.strip()

    ## Prompt for formatting tool call
    prompt = f"""
Consider the given context and question, then for each sentence in the context, determine whether it is relevant to answering the question. Provide a brief explanation of why each sentence is relevant or not before arriving at the verdict (Yes/No). Provide a final verdict for each sentence in order. Finally you MUST use the format_sentence_verdicts function to convert the sentences, your reasonings and your verdicts for each sentence into a valid JSON format.
Context: {context}
Question: {question}
Response:\n
    """.strip()

    message = [
        {"role": "user", "content": prompt}
    ]
    # verdicts_response_format = {
    #     "json_schema" : {
    #         "name" : "Sentences, Verdicts and Reasonings",
    #         "strict" : True,
    #         "schema" : {
    #             "type": "array",
    #             "items" : SentenceVerdict.model_json_schema()
    #         }
    #     }
    # }
    raw_response = call_llm(messages=message, tools=completeness_tools)
    ## Usage details for logging purposes:
    usage_details = raw_response.usage
    print(f"Token usage: completion_tokens={usage_details.completion_tokens} (reasoning_tokens={usage_details.completion_tokens_details.reasoning_tokens}), prompt_tokens={usage_details.prompt_tokens}, total_tokens={usage_details.total_tokens}, cached_tokens={usage_details.prompt_tokens_details.cached_tokens}\n")
    
    print("\n\n",raw_response,"\n\n\n")

    ## Structured output result extraction
    # response = raw_response.choices[0].message.content
    # response = json.loads(response)

    ## Formatting tool call result extraction
    tool_call = raw_response.choices[0].message.tool_calls[0]
    tool_result, tool_success = execute_completeness_tool_call(tool_call=tool_call)
    if tool_success:
        response = tool_result
    else:
        response = raw_response.choices[0].message.content

    # response_sentences = []
    # verdicts = []
    # reasonings = []
    # for obj in response:
    #     response_sentences.append(obj["sentence"])
    #     verdicts.append(True if "yes" in obj["verdict"].lower() else False)
    #     reasonings.append(obj["reasoning"])
    judgements = []
    for obj in response:
        judgements.append({
            "sentence": obj["sentence"],
            "reasoning": obj["reasoning"],
            "verdict": True if "yes" in obj["verdict"].lower() else False
        })
    verdicts = [obj["verdict"] for obj in judgements]

    num_relevant_sentences = sum(1 for v in verdicts if v)
    total_sentences = len(verdicts)
    score = (num_relevant_sentences / total_sentences) if total_sentences > 0 else 0

    return {
        "score": score,
        "judgements": judgements
    }




### ACTION DOCS PARSING AND RETRIEVAL

def get_oracle_actions(id_list, context : action_parsing.ActionParsingContext):
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
        parsed_actions.append(action_parsing.parse_action(action_string=content, context=context))

    return parsed_actions


def get_oracle_actions_as_str(id_list, context : action_parsing.ActionParsingContext):
    actions = get_oracle_actions(id_list=id_list, context=context)
    action_strings = []
    for action in actions:
        action_str = action_parsing.get_parsed_action_as_str(action=action)
        action_strings.append(action_str)
    full_str = "\n\n".join(action_strings)
    return full_str



### FULL METRIC EVALUATION PIPELINE


def evaluate_metric(metric_name, question, summary, action_ids_in_summary, oracle_ids, context : action_parsing.ActionParsingContext):
    # Getting the oracle actions
    oracle_actions = get_oracle_actions(id_list=oracle_ids, context=context)
    oracle_actions_str = "\n\n".join([action_parsing.get_parsed_action_as_str(action=action) for action in oracle_actions])

    # Getting the actions cited in the summary
    cited_actions = get_oracle_actions(id_list=action_ids_in_summary, context=context)
    cited_actions_str = "\n\n".join([action_parsing.get_parsed_action_as_str(action=action) for action in cited_actions])

    all_docs = oracle_actions + cited_actions
    all_docs_str = oracle_actions_str + "\n\n" + cited_actions_str

    summary_statements = get_statements(question=question, answer=summary)

    if metric_name == "faithfulness":
        result = faithfulness(question=question, summary=summary, docs=all_docs_str, statements=summary_statements)
    elif metric_name == "citation_correctness":
        result = citation_correctness(question=question, summary=summary, docs=all_docs_str, statements=summary_statements)
    elif metric_name == "relevance":
        result = relevance(question=question, summary=summary, statements=summary_statements)
    else:
        metric_name = None
        result = None
    
    full_result = {
        "metric": metric_name,
        "qu_summary": {
            "question": question,
            "all_relevant_action_ids": oracle_ids,
            "summary": summary,
            "action_ids_in_summary": action_ids_in_summary
        },
        "evaluation": result
    }

    return full_result



    
def main():
    logging.basicConfig(level=logging.INFO, filename="logfiles/own_ragas_metrics.log", format='%(asctime)s - %(levelname)s - %(message)s')

    context = action_parsing.ActionParsingContext(required_fields=["action_id", "action_title", "key_messages"])

    ## Loamy soils good answer
    # question = "What are the most effective ways to increase soil organic carbon on loamy soils?"
    # answer = "The most effective ways to increase soil organic carbon on loamy soils include growing cover crops, implementing reduced tillage practices, using crop rotation, applying organic amendments, and utilizing mixed organic-inorganic fertilizers.\n\nGrowing cover crops when fields are empty is particularly beneficial for increasing soil organic carbon on loamy soils (Action 898). Studies found increased soil carbon levels under cover crops, with further increases when legumes were included in the cover crop mix. Implementing reduced tillage or no-tillage practices significantly enhances soil organic carbon accumulation (Action 906). Twelve studies comparing no-tillage and conventionally tilled systems found consistently higher soil organic carbon in soils under reduced tillage systems, and the effectiveness is further enhanced when combined with cover cropping and manure application. Using crop rotation, especially when legumes are included, also proves beneficial (Action 857). Four studies found increased soil organic carbon under crop rotations, particularly when legumes were incorporated into the system. Applying mixed organic and inorganic amendments provides another effective approach (Action 902). Four controlled trials found more organic carbon in soils treated with mixed fertilizers compared to inorganic fertilizers alone. Additionally, applying manures and agricultural composts can increase soil carbon levels (Action 911), though this method requires careful consideration of potential trade-offs. Finally, formulated chemical compounds like nitrogen or phosphorus fertilizers can also contribute to soil organic matter increases (Action 909), with five of six studies showing increased soil organic matter when these compounds were applied to various soil types including loam."
    # action_ids_in_answer = ["898","906","857","902","911","909"]
    # oracle_ids = ["906","857","902","907","911"]

    ## Bears bad answer
    question = "What are the most beneficial actions for reducing human-wildlife conflict with bears?"
    answer = "The most effective bear conflict reduction strategies include deterrence techniques and preventing access to food sources, while translocation shows mixed results.\nScaring or deterring bears using projectiles, noisemakers, guard dogs, or unpleasant substances has proven beneficial in modifying bear behavior and reducing conflicts in human-occupied areas (ref: 2347). Preventing bears from accessing anthropogenic food sources like garbage, crops, and pet food through bear-proof containers or exclusion methods is also likely beneficial for conflict reduction (ref: 2346). Conditioned taste aversion, which involves adding illness-inducing agents to problem foods at non-residential sites like orchards or campsites, shows promise in creating food aversions (ref: 2384). Enforcement measures for bear-proof garbage disposal demonstrate unknown effectiveness due to limited evidence (ref: 2345). Translocation of habituated bears is less recommended because it often leads to trade-offs - bears may return to conflict sites or re-offend after relocation (ref: 2341)."
    action_ids_in_answer = ['2341', '2345', '2346', '2347', '2384']
    oracle_ids = ["2330","2336","2346","2347","2385"]
    # statements = [
    #     'The most effective bear conflict reduction strategies include deterrence techniques.', 
    #     'The most effective bear conflict reduction strategies include preventing access to food sources.', 
    #     'Translocation shows mixed results for reducing bear conflict.', 
    #     'Scaring or deterring bears using projectiles, noisemakers, guard dogs, or unpleasant substances has proven beneficial in modifying bear behavior.', 
    #     'Scaring or deterring bears has proven beneficial in reducing conflicts in human-occupied areas.', 
    #     'Preventing bears from accessing anthropogenic food sources like garbage, crops, and pet food is likely beneficial for conflict reduction.', 
    #     'Preventing access to food sources can be done through bear-proof containers or exclusion methods.', 
    #     'Conditioned taste aversion shows promise in creating food aversions in bears.', 
    #     'Conditioned taste aversion involves adding illness-inducing agents to problem foods at non-residential sites like orchards or campsites.', 
    #     'Enforcement measures for bear-proof garbage disposal have unknown effectiveness due to limited evidence.', 
    #     'Translocation of habituated bears is less recommended because it often leads to trade-offs.', 
    #     'Translocated bears may return to conflict sites.', 
    #     'Translocated bears may re-offend after relocation.'
    # ]
    # statements = [
    #     "The most effective bear conflict reduction strategies include deterrence techniques and preventing access to food sources, while translocation shows mixed results.",
    #     "Scaring or deterring bears using projectiles, noisemakers, guard dogs, or unpleasant substances has proven beneficial in modifying bear behavior and reducing conflicts in human-occupied areas.",
    #     "Preventing bears from accessing anthropogenic food sources like garbage, crops, and pet food through bear-proof containers or exclusion methods is also likely beneficial for conflict reduction.",
    #     "Conditioned taste aversion, which involves adding illness-inducing agents to problem foods at non-residential sites like orchards or campsites, shows promise in creating food aversions.",
    #     "Enforcement measures for bear-proof garbage disposal demonstrate unknown effectiveness due to limited evidence.",
    #     "Translocation of habituated bears is less recommended because it often leads to trade-offs - bears may return to conflict sites or re-offend after relocation."
    # ]

    ## Bears good summary
    # question = "What are the most beneficial actions for reducing human-wildlife conflict with bears?"
    # answer = "Evidence indicates several actions can reduce bear-related conflicts. Diversionary feeding reduced nuisance behaviour by black bears in two before-and-after studies, and brown bears in Slovenia obtained 22\u201363% of annual dietary energy from provided food (2323). Scaring/deterrence had mixed outcomes: some studies found noise/pain deterrents did not prevent black bears returning to urban or human-occupied sites, while other studies found such actions deterred bears from seeking food; chasing nuisance black bears with dogs caused them to stay away longer; an electric fence prevented polar bear entry to a compound; chemical and acoustic repellents did not deter polar bears from baits in most cases (2347). Preventing access to food sources with electric shock devices stopped American black bears from accessing or damaging bird feeders (2346). Conditioned taste aversion led black bears to avoid treated foods (2384). Issuing enforcement notices requiring appropriate dumpster use did not reduce garbage accessibility to black bears (2345). Translocating problem or habituated bears often resulted in bears returning to capture locations and/or continuing nuisance, and for grizzly and black bears reduced survival compared to non-translocated bears; however, one controlled study found translocated brown bears occurred less frequently inside high potential conflict areas than non-translocated bears (2336, 2341)."
    # action_ids_in_answer = ["2323","2336","2341","2345","2346","2347","2384"]
    # oracle_ids = ["2330","2336","2346","2347","2385"]
    # statements = [
    #     'Evidence indicates several actions can reduce bear-related conflicts.',
    #     'Diversionary feeding reduced nuisance behaviour by black bears in two before-and-after studies.',
    #     'Brown bears in Slovenia obtained 22–63%% of their annual dietary energy from provided food.',
    #     'Scaring and deterrence methods had mixed outcomes.', 
    #     'Some studies found that noise and pain deterrents did not prevent black bears from returning to urban or human-occupied sites.', 
    #     'Other studies found that scaring and deterrence actions deterred bears from seeking food.',
    #     'Chasing nuisance black bears with dogs caused them to stay away longer.',
    #     'An electric fence prevented a polar bear from entering a compound.', 
    #     'In most cases, chemical and acoustic repellents did not deter polar bears from baits.', 
    #     'Preventing access to food sources with electric shock devices stopped American black bears from accessing or damaging bird feeders.',
    #     'Conditioned taste aversion led black bears to avoid treated foods.', 
    #     'Issuing enforcement notices that required appropriate dumpster use did not reduce garbage accessibility for black bears.',
    #     'Translocating problem or habituated bears often resulted in the bears returning to their capture locations and/or continuing their nuisance behavior.',
    #     'Translocation reduced the survival of grizzly and black bears compared to non-translocated bears.', 
    #     'One controlled study found that translocated brown bears occurred less frequently inside high-potential conflict areas than non-translocated bears.'
    # ]

    ## Intentionally bad (partially irrelevant) answer generated to soils query
    # answer = "Evidence suggests several management practices influence soil properties on loamy soils. Adopting no-tillage or reduced tillage practices consistently resulted in the stratification and accumulation of organic carbon near the soil surface compared to conventional tillage, although effects on total carbon in the deeper soil profile were mixed (1114). The application of lime to acidic loamy soils successfully raised pH into the optimal range for common row crops, which improved nutrient availability but did not alter the soil's water holding capacity (1131). The inclusion of cover crops in rotations was found to increase soil organic carbon stocks in the topsoil in multiple meta-analyses, though the magnitude of effect varied by climate and species (1102). Precision irrigation technologies, such as drip systems, significantly reduced water consumption by 30-50% compared to furrow irrigation in arid and semi-arid regions, improving water use efficiency (1120). Direct application of organic amendments like compost or animal manure was a highly effective strategy, with one long-term study showing a doubling of soil carbon content over 20 years in treated plots (1125). Subsurface tile drainage systems were shown to improve soil aeration and trafficability in poorly drained loams, but had little effect on phosphorus leaching (1105)."


    all_ids = list(set(action_ids_in_answer) | set(oracle_ids))

    docs_str = get_oracle_actions_as_str(id_list=all_ids, context=context)
    docs = get_oracle_actions(id_list=all_ids, context=context)
    # print(faithfulness(question=question, summary=answer, docs=docs_str))

    result = completeness(question=question, summary=answer, docs=docs)

    print(f"'score':{result['score']}")
    # print(f"\n'question': {question}")
    print(f"\n'statements':")
    for s in result["statements"]:
        print(f'"{s}"')
    # print(f"\n'citations':")
    # for c in result["citations"]:
    #     print(f'{c}')
    print(f"\n'verdicts': {result["verdicts"]}")
    print(f"\n'reasonings':")
    for r in result["reasonings"]:
        print(f'"{r}"')

    # statements = get_statements(question=question, answer=answer)
    # for s in statements:
    #     print(f'"{s}"')

    # result = answer_relevance(query=question, answer=answer, n=10)
    # print(f"'score':{result['score']}")
    # for q in result["questions"]:
    #     print(f'"{q}"')


if __name__ == "__main__":
    main()





