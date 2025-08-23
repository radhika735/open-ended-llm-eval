import os
import json
import re
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI

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


# copied from "Evaluation of RAG Metrics for Question Answering in the Telecom Domain" paper
def get_ragas_result(question, answer):
    ragas_result = defaultdict(dict)

    prompt = f"""
        Given a question and answer, create one or more statements from each sentence in the given answer. Output the statements in the following format strictly. [Statement n]: ... where the n is the statement number.\nquestion: {question}\nanswer: {answer}\nStatements:\n
    """.strip()

    client = get_client()
    statements_response = client.chat.completions.create(
        model="o1",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    # Splitting statements
    response = re.sub(
        r"^.*?[0-9]+\]*[:\.]\s(.*?)", r"\1", response, flags=re.MULTILINE
    ).strip()
    statements = sent_tokenize(response)
    # key_idf = f"({context_id},{ground_truth_id})"
    # ragas_result[key_idf]["faithfulnessStatements"] = statements
    eval = {
        "faithfulnessStatements": statements,

    }

    ragas_result = {
        "question": question, 
        "answer": answer,
        "eval": eval
    }

