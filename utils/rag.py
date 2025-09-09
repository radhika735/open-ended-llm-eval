import os
import Stemmer
import logging
import bm25s
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.action_parsing import ActionParsingContext, get_all_parsed_actions, get_parsed_action_as_str, get_parsed_action_metadata


def sparse_retrieve_docs(query_string, context : ActionParsingContext, k=3, offset=0):
    """
    Perform a sparse retrieval of documents based on a query string.
    
    Args:
        query_string (str): The search query for finding relevant actions
        context (ActionParsingContext): The context for action retrieval (e.g can use this to find the user's set doc_type, required_fields, metadata_fields)
        k (int): Number of top results to return (default: 3)
        offset (int): Number of results to skip (default: 0, used for pagination)
    
    Returns:
        list: Top k action documents matching the query, starting from offset
    """
    parsed_actions = get_all_parsed_actions(context=context)
    corpus = []
    for action in parsed_actions:
        action_string = get_parsed_action_as_str(action=action)
        corpus.append(action_string)

    stemmer = Stemmer.Stemmer("english")

    # Tokenize the corpus and index it
    logging.debug("Tokenizing and indexing the corpus...")
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

    retriever = bm25s.BM25()
    logging.debug("Creating BM25 retriever...")
    retriever.index(corpus_tokens)

    logging.debug("BM25 retriever is ready.")
    
    # Search the corpus with the provided query
    # Retrieve more results than needed to handle offset
    total_results_needed = k + offset
    query_tokens = bm25s.tokenize(query_string, stopwords="en", stemmer=stemmer)
    logging.debug(f"Searching for query: {query_string}")
    docs, scores = retriever.retrieve(query_tokens, k=total_results_needed)
    
    logging.debug(f"Docs: {docs}\nScores: {scores} (type: {type(scores)})")

    # Format results with metadata, applying offset and limit
    results = []
    for i, (doc, score) in enumerate(zip(docs[0], scores[0])):
        # Skip results before offset
        if i < offset:
            continue
        # Stop if we've collected enough results
        if len(results) >= k:
            break
            
        score = score.item()
        logging.debug(f"doc: {doc}")
        logging.debug(f"score: {score}, type: {type(score)}")
        action = parsed_actions[doc]
        result = get_parsed_action_metadata(action=action, context=context)
        result["rank"] = i + 1  # Keep original rank from search
        results.append(result)
    
    logging.debug(f"Found {len(results)} documents for query '{query_string}':")
    return results



def get_dense_embeddings(all_docs, context : ActionParsingContext, load_from_cache=True, save_to_cache=True, np_cache_dir="answer_gen_data/retrieval_methods/dense_embeddings_cached"):
    doc_type = context.get_doc_type()
    np_cache_file = os.path.join(np_cache_dir, f"{doc_type}_nomic_dense_embeddings.npy")
    if load_from_cache and os.path.exists(np_cache_file):
        return np.load(np_cache_file)
    else:
        import utils.create_dense_embeddings_actions as create_dense_embeddings_actions
        embeddings = create_dense_embeddings_actions.get_embeddings(docs=all_docs, save_to_cache=save_to_cache, cache_file=np_cache_file) # will take half an hour to run
        return embeddings



def dense_retrieve_docs(query_string, context : ActionParsingContext, k=3, offset=0):
    """
    Perform a dense retrieval of documents based on a query string.

    Args:
        query_string (str): The search query for finding relevant actions
        context (ActionParsingContext): The context for action retrieval (e.g can use this to find the user's set doc_type, required_fields, metadata_fields)
        k (int): Number of top results to return (default: 3)
        offset (int): Number of results to skip (default: 0, used for pagination)

    Returns:
        list: Top k action documents matching the query, starting from offset
    """
    parsed_actions = get_all_parsed_actions(context=context)
    corpus = []
    for action in parsed_actions:
        action_string = get_parsed_action_as_str(action=action)
        corpus.append(action_string)

    # Encode the query
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    query_embedding = model.encode(query_string, normalize_embeddings=True)
    # Get the document embeddings
    dense_embeddings = get_dense_embeddings(all_docs=corpus, context=context)

    # Compute cosine similarities
    similarities = cosine_similarity(query_embedding, dense_embeddings)[0]
    # Get the top k results
    k = min(k, len(similarities))
    top_results = torch.topk(similarities, k = k + offset)

    results = []
    for rank, idx in enumerate(top_results.indices):
        if rank < offset:
            continue
        action = parsed_actions[idx]
        result = get_parsed_action_metadata(action=action, context=context)
        result["rank"] = rank + 1
        results.append(result)

    return results



def reciprocal_rank_fusion(ranked_lists, num_docs, dampener=60):
    all_docs = {}
    scores = defaultdict()

    for r_list in ranked_lists:
        for doc in r_list:
            all_docs[doc["action_id"]] = doc
            new_score = 1 / (doc["rank"] + dampener)
            scores[doc["action_id"]] += new_score

    # Get the top k (num_docs) results
    top_ids = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:num_docs]
    top_docs = [all_docs[doc_id] for doc_id, _ in top_ids]
    return top_docs



def cross_encoder_scores(query, docs):
    model_name = "cross_encoder/ms-marco-MiniLm-L-6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    features = tokenizer([query] * len[docs], docs, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        scores = model(**features).logits
    return scores.tolist()



# NEED TO DOUBLE CHECK THIS
def hybrid_retrieve_docs(query_string, context : ActionParsingContext, fusion_type = "cross-encoder",  k=3, offset=0):
    sparse_docs = sparse_retrieve_docs(query_string=query_string, context=context, k=k, offset=offset)
    dense_docs = dense_retrieve_docs(query_string=query_string, context=context, k=k, offset=offset)

    # Using reciprocal rank fusion to rerank the documents
    if fusion_type == "reciprocal rank fusion":
        combined_results = reciprocal_rank_fusion([sparse_docs, dense_docs], num_docs=k)
        return combined_results
    elif fusion_type == "cross-encoder":
        # Using a cross-encoder:
        candidate_docs = list(set(sparse_docs).union(set(dense_docs)))
        scores = cross_encoder_scores(query_string, candidate_docs)
        flattened_scores = [score[0] for score in scores]

        if len(scores) >= 2:
            top_docs = [candidate_docs[i] for i in np.argsort(flattened_scores)[-2:][::-1]]
        else:
            top_docs = candidate_docs[:len(flattened_scores)]

        return top_docs
    else:
        logging.error("Invalid fusion type given to hybrid document retriever in utils/action_retrieval.hybrid_retrieve_docs")
        return []



def main():
    logging.basicConfig(filename="logfiles/action_retrieval.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    context = ActionParsingContext(
        required_fields=["action_id", "action_title", "key_messages"]
    )

    ## Testing parsed actions with bg km
    docs = get_all_parsed_actions(context=context)
    for i in range(100, 105):
        print(docs[i])




if __name__ == "__main__":
    main()