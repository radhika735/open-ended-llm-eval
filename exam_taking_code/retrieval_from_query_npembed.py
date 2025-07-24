import time
import json
import os
import random
import re
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import csv 
import logging


logging.basicConfig(filename = "logfiles/retrieval_from_query_npembed.log", level=logging.DEBUG)


def load_all_documents(source="preparation/key_messages/km_all_textfiles", use_cache = True, cache_file = "examtaking/km_docs_cache.json"):

    """Storing documents in required format for future use. """

    if use_cache and os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            cached_data = json.load(f)
            return cached_data['docs'], cached_data['tokenised_docs']
    
    docs = []
    tokenised_docs = []

    for entry in os.scandir(source):
        with open(entry.path, 'r', encoding='utf-8') as file:
            content = file.read()
            docs.append(content)
            tokenised_docs.append(content.split())
    
    if use_cache:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({'docs': docs, 'tokenised_docs': tokenised_docs}, f)
    
    return docs, tokenised_docs


def get_dense_embeddings(all_docs, dense_model, np_cache_file="retrievalmethods/dense_embeddings_cached/km_nomic_dense_embeddings.npy"):
    if os.path.exists(np_cache_file):
        return np.load(np_cache_file)
    
    dense_embeddings = []
    batch_size = 32

    for i in range(0, len(all_docs), batch_size): #This will take around an hour to run for new dense_models
        batch = all_docs[i:i+batch_size]
        for doc in batch:
            doc_embedding = dense_model.encode(doc)
            dense_embeddings.append(doc_embedding)

    dense_embeddings = np.array(dense_embeddings)  #4000 docs - using numpy array rather than vector database
    np.save(np_cache_file, dense_embeddings)
    return dense_embeddings


def dense_retrieve_doc_indices(question, dense_model, dense_embeddings, all_docs):
    query_embedding = dense_model.encode([question['question']], batch_size=32)
    similarities = cosine_similarity(query_embedding, dense_embeddings)[0]
    k = 4
    k = min(k, len(similarities))
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    return top_k_indices


def sparse_retrieve_doc_indices(question, sparse_model, all_docs):
    k = 4
    query_tokens = question['question'].split()
    scores = sparse_model.get_scores(query_tokens)
    top_k_indices = np.argsort(scores)[-k:][::-1]
    return top_k_indices


def get_cross_encoder(model_name):
    tokeniser = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokeniser, model


def get_cross_encoder_scores(question, relevant_docs, model_name):
    tokeniser, cross_encoder_model = get_cross_encoder(model_name)
    features = tokeniser([question['question']] * len(relevant_docs), relevant_docs, truncation=True, padding=True, return_tensors='pt')
    with torch.no_grad():
        outputs = cross_encoder_model(**features).logits
    return outputs.tolist()


def hybrid_retrieve_docs(question, dense_embeddings, all_docs, dense_model, sparse_model, cross_encoder_model):
    dense_retrieved_action_indices = dense_retrieve_doc_indices(question, dense_model, dense_embeddings, all_docs)
    sparse_retrieved_action_indices = sparse_retrieve_doc_indices(question, sparse_model, all_docs)

    combined_action_indices = set(dense_retrieved_action_indices) | set(sparse_retrieved_action_indices)
    combined_docs = [all_docs[i] for i in combined_action_indices]
    combined_scores = get_cross_encoder_scores(question, combined_docs, cross_encoder_model)
    flattened_scores = [score[0] for score in combined_scores]

    sorted_indices = np.argsort(flattened_scores)[-4:][::-1]
    final_retrieved_docs = [combined_docs[i] for i in sorted_indices]

    return final_retrieved_docs



def main():
    query = {
        "question": "Which conservation actions are the most cost-effective at increasing biodiversity in grasslands?"
    }

    all_docs, tokenised_docs = load_all_documents()

    dense_model = SentenceTransformer("nomic-ai.nomic-embed-text-v1.5", trust_remote_code=True)
    sparse_model = BM25Okapi(tokenised_docs)
    cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    dense_embeddings = get_dense_embeddings(all_docs, dense_model)

    retrieved_docs = hybrid_retrieve_docs(query, dense_embeddings, all_docs, dense_model, sparse_model, cross_encoder_model)


if __name__ == "__main__":
    main()