from multiprocessing import context
import os
import json
import re
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
from nltk.tokenize import sent_tokenize

load_dotenv()


# API Configuration
def get_client():
    """
    Get an OpenAI client configured for openrouter.
    
    Returns:
        OpenAI: Configured client
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required for OpenRouter")
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def call_llm(messages, model="google/gemini-2.5-pro"):
    client = get_client()
    response = client.chat.completions.create(
        model = model,
        messages = messages
    )
    return response


# copied heavily from "Evaluation of RAG Metrics for Question Answering in the Telecom Domain" paper
def get_statements(question, answer):
    prompt = f"""
        Given a question and answer, create one or more statements from each sentence in the given answer. Output the statements in the following format strictly. [Statement n]: ... where the n is the statement number.\nquestion: {question}\nanswer: {answer}\nStatements:\n
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


# copied heavily from "Evaluation of RAG Metrics for Question Answering in the Telecom Domain" paper
def faithfulness(question, answer):
    statements = get_statements(question, answer)
    verdicts = []
    reasonings = []
    for idx, statement in enumerate(statements):
        prompt = f"""
            Consider the given context and following statement, then determine whether it is supported by the information present in the context. Provide a brief explanation before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order at the end in the given format. Do not deviate from the specified format.\nContext: {context}\nStatement: {statement}\nResponse:\n
        """.strip()
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        response = call_llm(messages=messages).choices[0].message.content
        response = response.strip()
        verdict = "yes" in response.lower()
        verdicts.append(verdict)
        reasonings.append(response)

    num_supported_statements = sum(1 for v in verdicts if v)
    total_statements = len(verdicts)
    score = (num_supported_statements / total_statements) if total_statements > 0 else 0

    return {
        "score":score,
        "verdicts":verdicts,
        "reasonings":reasonings
    }


# copied heavily from "Evaluation of RAG Metrics for Question Answering in the Telecom Domain" paper
def answer_relevance(query, answer, n=10):
    questions = []
    prompt = f"""
        Generate {n} potential questions for the given answer. Output the questions in the following format strictly. [Question n]: ... where the n is the question number.\nanswer: {answer}\nQuestions:\n
    """.strip()
    messages = [
        {
            "role": "user",
            "content":prompt
        }
    ]
    response = call_llm().choices[0].message.content
    response = response.strip()
    parsed_questions = re.sub(
        r"^.*?[0-9]+\]*[:\.]\s(.*?)", r"\1", response, flags=re.MULTILINE
    ).strip()
    tokenised_questions = sent_tokenize(parsed_questions)








