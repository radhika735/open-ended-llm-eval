import json
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import logging


logging.basicConfig(filename = "logfiles/retrieval_from_query_npembed.log", level=logging.DEBUG)

ACTIONS_TYPE = "km_evid"# other options are "km_noevid" and "km_all"


def load_all_documents(source = f"action_data/key_messages/{ACTIONS_TYPE}", use_cache = True, cache_file = f"exam_taking_data/{ACTIONS_TYPE}_docs_cache.json"):

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



def load_cached_retrieval_results(retrieval_type, result_type):
    # retrieval_type can be 'dense', 'sparse', or 'hybrid'.
    # result_type can be 'docs' or 'indices'. Where 'indices' contains for each query, the relevant all_docs indices and action numbers.
    cache_dir = f"exam_taking_data/retrieval_methods/cached_retrieval_{result_type}"
    cache_file = os.path.join(cache_dir, f"{retrieval_type}_retrieval_cache.json")
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                logging.debug(f"Error loading {result_type} cache for {retrieval_type}. New cache needed.")
                return {}
    return {}


### NEED TO TEST
def cache_new_retrieval_results(query, results, retrieval_type, result_type):
    cache_dir="exam_taking_data/retrieval_methods/cached_retrieval_{result_type}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{retrieval_type}_retrieval_cache.json")
    cache = load_cached_retrieval_results(retrieval_type, result_type)
    if query in cache:
        if result_type == "indices":
            cache[query].update(results)# important if updating cached_retrieval_indices/... 
                                        # because in this case cache.update({query : results}) overwrite would delete  
                                        # any existing results for different source dirs (km_all, km_evid, km_noevid).
        else:
            cache.update({query: results})    
        logging.debug(f"Updating existing cache for query: {query}")
    else:
        logging.debug(f"Caching new results for query: {query}")
        cache.update({query : results})

    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)



def get_action_nums_from_indices(indices, all_docs):
    relevant_docs = [all_docs[i] for i in indices]
    action_num_matches = [re.search(r"^(\d+):", doc) for doc in relevant_docs]
    action_nums = []
    for match in action_num_matches:
        if match:
            action_nums.append(int(match.group(1)))
    return action_nums
    
    

def get_dense_embeddings(all_docs, dense_model, np_cache_file="exam_taking_data/retrieval_methods/dense_embeddings_cached/km_nomic_dense_embeddings.npy"):
    if os.path.exists(np_cache_file):
        return np.load(np_cache_file)
    
    dense_embeddings = []
    batch_size = 32

    for i in range(0, len(all_docs), batch_size): # This will take around an hour to run for new dense_models
        batch = all_docs[i:i+batch_size]
        for doc in batch:
            doc_embedding = dense_model.encode(doc)
            dense_embeddings.append(doc_embedding)

    dense_embeddings = np.array(dense_embeddings)  # 4000 docs - using numpy array rather than vector database
    np.save(np_cache_file, dense_embeddings)
    return dense_embeddings



### NEED TO TEST
def dense_retrieve_doc_indices(query, dense_model, dense_embeddings, all_docs, use_cache=True):
    if use_cache:
        cache = load_cached_retrieval_results("dense", "indices")
        if query in cache:
            if f"{ACTIONS_TYPE}" in cache[query]:
                return cache[query][f"{ACTIONS_TYPE}"]["indices"]
        
    query_embedding = dense_model.encode([query], batch_size=32)
    similarities = cosine_similarity(query_embedding, dense_embeddings)[0]
    k = 4
    k = min(k, len(similarities))
    top_k_indices = np.argsort(similarities)[-k:][::-1]

    if use_cache:
        action_nums = get_action_nums_from_indices(top_k_indices, all_docs)
        cache_new_retrieval_results(query, {f"{ACTIONS_TYPE}": {"indices":top_k_indices.tolist(), "action_nums":action_nums}}, "dense", "indices")
    return top_k_indices



### NEED TO TEST
def sparse_retrieve_doc_indices(query, sparse_model, all_docs, use_cache=True):
    if use_cache:
        cache = load_cached_retrieval_results("sparse", "indices")
        if query in cache:
            if f"{ACTIONS_TYPE}" in cache[query]:
                return cache[query][f"{ACTIONS_TYPE}"]["indices"]
            
    k = 4
    query_tokens = query.split()
    scores = sparse_model.get_scores(query_tokens)
    top_k_indices = np.argsort(scores)[-k:][::-1]

    if use_cache:
        action_nums = get_action_nums_from_indices(top_k_indices, all_docs)
        cache_new_retrieval_results(query, {f"{ACTIONS_TYPE}": {"indices":top_k_indices.tolist(), "action_nums":action_nums}}, "sparse", "indices")
    return top_k_indices



def get_cross_encoder(model_name, cache_dir = "retrieval_methods/cached_models"):
    os.makedirs(cache_dir, exist_ok = True)
    tokeniser = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
    return tokeniser, model



def get_cross_encoder_scores(query, relevant_docs, model_name):
    tokeniser, cross_encoder_model = get_cross_encoder(model_name)
    features = tokeniser([query] * len(relevant_docs), relevant_docs, truncation=True, padding=True, return_tensors='pt')
    with torch.no_grad():
        outputs = cross_encoder_model(**features).logits
    return outputs.tolist()



def hybrid_retrieve_docs(query, dense_embeddings, all_docs, dense_model, sparse_model, cross_encoder_model, use_cache=True):
    if use_cache:
        cache = load_cached_retrieval_results("hybrid", "docs")
        if query in cache:
            return cache[query]

    dense_retrieved_action_indices = dense_retrieve_doc_indices(query, dense_model, dense_embeddings, all_docs, use_cache)
    sparse_retrieved_action_indices = sparse_retrieve_doc_indices(query, sparse_model, all_docs, use_cache)

    combined_action_indices = set(dense_retrieved_action_indices) | set(sparse_retrieved_action_indices)
    combined_docs = [all_docs[i] for i in combined_action_indices]
    combined_scores = get_cross_encoder_scores(query, combined_docs, cross_encoder_model)
    flattened_scores = [score[0] for score in combined_scores]

    sorted_indices = np.argsort(flattened_scores)[-4:][::-1]
    final_retrieved_docs = [combined_docs[i] for i in sorted_indices]

    if use_cache:
        action_nums = get_action_nums_from_indices(sorted_indices, all_docs)
        cache_new_retrieval_results(query, {f"{ACTIONS_TYPE}":{"indices":sorted_indices.tolist(), "action_nums":action_nums}}, "hybrid", "indices")
        cache_new_retrieval_results(query, final_retrieved_docs, "hybrid", "docs")

    return final_retrieved_docs



def main():
    query = "Which conservation actions are the most cost-effective at increasing biodiversity in grasslands?"

    all_docs, tokenised_docs = load_all_documents()

    dense_model = SentenceTransformer("nomic-ai.nomic-embed-text-v1.5", trust_remote_code=True)
    sparse_model = BM25Okapi(tokenised_docs)
    cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    dense_embeddings = get_dense_embeddings(all_docs, dense_model)

    retrieved_docs = hybrid_retrieve_docs(query, dense_embeddings, all_docs, dense_model, sparse_model, cross_encoder_model)



if __name__ == "__main__":
    main()