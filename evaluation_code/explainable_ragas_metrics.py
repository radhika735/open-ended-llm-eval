# copied heavily from "Evaluation of RAG Metrics for Question Answering in the Telecom Domain" paper
from multiprocessing import context
import os
import json
import logging
import re
from dotenv import load_dotenv
from openai import OpenAI
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils.action_retrieval import ActionRetrievalContext, parse_action, get_parsed_action_as_str

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


def faithfulness(question, answer, docs):
    statements = get_statements(question, answer)
    verdicts = []
    reasonings = []
    for idx, statement in enumerate(statements):
        prompt = f"""
            Consider the given context and following statement, then determine whether it is supported by the information present in the context. Provide a brief explanation before arriving at the verdict (Yes/No). Provide a final verdict for each statement in order at the end in the given format. Do not deviate from the specified format.\nContext: {docs}\nStatement: {statement}\nResponse:\n
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


def embed(text, model_name="nomic-ai/nomic-embed-text-v1.5"):
    model = SentenceTransformer(model_name, trust_remote_code=True)
    embeddings = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
    return embeddings


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
    parsed_response = re.sub(
        r"^.*?[0-9]+\]*[:\.]\s(.*?)", r"\1", response, flags=re.MULTILINE
    ).strip()
    gen_questions = sent_tokenize(parsed_response)

    original_qu_embedding = embed(query)
    gen_qus_embeddings = embed(gen_questions)

    answer_relevance_scores = []
    for gen_embedding in gen_qus_embeddings:
        sim_score = cosine_similarity(original_qu_embedding.reshape(1, -1), gen_embedding.reshape(1, -1))
        answer_relevance_scores.append(sim_score)
    
    avg_score = np.mean(answer_relevance_scores) if answer_relevance_scores > 0 else 0
    avg_score = float(avg_score)
    return avg_score


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

    
def main():
    logging.basicConfig(level=logging.INFO, filename="logfiles/explainable_ragas_metrics.log", format='%(asctime)s - %(levelname)s - %(message)s')

    context = ActionRetrievalContext(required_fields=["action_id", "action_title", "key_messages"])

    question = "What are the most effective ways to increase soil organic carbon on loamy soils?"
    answer = "The most effective ways to increase soil organic carbon on loamy soils include growing cover crops, implementing reduced tillage practices, using crop rotation, applying organic amendments, and utilizing mixed organic-inorganic fertilizers.\n\nGrowing cover crops when fields are empty is particularly beneficial for increasing soil organic carbon on loamy soils (Action 898). Studies found increased soil carbon levels under cover crops, with further increases when legumes were included in the cover crop mix. Implementing reduced tillage or no-tillage practices significantly enhances soil organic carbon accumulation (Action 906). Twelve studies comparing no-tillage and conventionally tilled systems found consistently higher soil organic carbon in soils under reduced tillage systems, and the effectiveness is further enhanced when combined with cover cropping and manure application. Using crop rotation, especially when legumes are included, also proves beneficial (Action 857). Four studies found increased soil organic carbon under crop rotations, particularly when legumes were incorporated into the system. Applying mixed organic and inorganic amendments provides another effective approach (Action 902). Four controlled trials found more organic carbon in soils treated with mixed fertilizers compared to inorganic fertilizers alone. Additionally, applying manures and agricultural composts can increase soil carbon levels (Action 911), though this method requires careful consideration of potential trade-offs. Finally, formulated chemical compounds like nitrogen or phosphorus fertilizers can also contribute to soil organic matter increases (Action 909), with five of six studies showing increased soil organic matter when these compounds were applied to various soil types including loam."
    action_ids_in_answer = ["898","906","857","902","911","909"]
    oracle_ids = ["906","857","902","907","911"]
    all_ids = list(set(action_ids_in_answer) | set(oracle_ids))

    docs = get_oracle_actions_as_str(id_list=all_ids, context=context)
    print(faithfulness(question=question, answer=answer, docs=docs))


if __name__ == "__main__":
    main()





