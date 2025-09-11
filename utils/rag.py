import os
import Stemmer
import logging
import bm25s
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
# from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils.action_parsing import ActionParsingContext, get_all_parsed_actions, get_parsed_action_as_str, get_parsed_action_metadata, get_parsed_action_by_id

CROSS_ENCODER_MODEL = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
DENSE_MODEL = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
BM25_RETRIEVER = None
BM25_CORPUS = None


def get_bm25_retriever(context: ActionParsingContext):
    global BM25_RETRIEVER, BM25_CORPUS
    if BM25_RETRIEVER is not None:
        return BM25_RETRIEVER, BM25_CORPUS

    parsed_actions = get_all_parsed_actions(context=context, load_from_all_cache=True, save_to_all_cache=True, saved_to_separated_cache=True)
    corpus = [get_parsed_action_as_str(action=a) for a in parsed_actions]

    stemmer = Stemmer.Stemmer("english")
    corpus_tokens = bm25s.tokenize(corpus, stopwords="en", stemmer=stemmer)

    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    BM25_RETRIEVER = retriever
    BM25_CORPUS = parsed_actions
    return retriever, parsed_actions


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
    # Get cached BM25 retriever and full parsed_actions corpus:
    retriever, parsed_actions = get_bm25_retriever(context=context)
    # Tokenize the query:
    stemmer = Stemmer.Stemmer("english")
    query_tokens = bm25s.tokenize(query_string, stopwords="en", stemmer=stemmer)
    
    # Search the corpus with the provided query:
    # Retrieve more results than needed to handle offset:
    total_results_needed = k + offset
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


def get_dense_embeddings(all_docs, context : ActionParsingContext, load_from_cache=True, save_to_cache=True, np_cache_dir="summary_gen_data/all_actions_dense_embeddings_cache"):
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
    corpus = [get_parsed_action_as_str(action=action) for action in parsed_actions]

    # Encode the query
    query_embedding = DENSE_MODEL.encode(query_string, normalize_embeddings=True)
    # Get the document embeddings
    dense_embeddings = get_dense_embeddings(all_docs=corpus, context=context)

    # Compute cosine similarities
    similarities = torch.tensor(cosine_similarity(query_embedding.reshape(1, -1), dense_embeddings)[0])
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
    all_docs = {} # is a dict storing action_id -> doc mapping.
    new_scores = defaultdict(int) # is a dict storing action_id -> new_rank mapping.

    for r_list in ranked_lists:
        for doc in r_list:
            doc_id = doc["action_id"]
            # Store the doc without the rank field (to avoid confusion):
            rankless_doc = {doc_key: doc[doc_key] for doc_key in doc if doc_key != "rank"}
            all_docs[doc_id] = rankless_doc
            
            new_score = 1 / (doc["rank"] + dampener)
            new_scores[doc_id] += new_score

    # Get the top k (num_docs) results
    top_ids_and_scores = sorted(new_scores.items(), key=lambda x: x[1], reverse=True)[:num_docs]
    
    results = []
    for rank, (id, score) in enumerate(top_ids_and_scores):
        result = all_docs[id]
        result["rank"] = rank + 1
        results.append(result)
    
    return results


def cross_encoder_scores(query, docs):
    pairs = [(query, get_parsed_action_as_str(action=doc)) for doc in docs]
    scores = CROSS_ENCODER_MODEL.predict(pairs)
    return scores.tolist()


def hybrid_retrieve_docs(query_string, context : ActionParsingContext, fusion_type = "cross-encoder",  k=3, offset=0):
    # sparse_retrieve_docs and dense_retrieve_docs return lists of actions metadata (parsed into a dictionary), with each action dict also having a "rank" field
    sparse_docs = sparse_retrieve_docs(query_string=query_string, context=context, k=k+offset, offset=0)
    dense_docs = dense_retrieve_docs(query_string=query_string, context=context, k=k+offset, offset=0)

    if fusion_type == "reciprocal rank fusion":# Using reciprocal rank fusion to rerank the documents.
        combined_results = reciprocal_rank_fusion([sparse_docs, dense_docs], num_docs=k)
        return combined_results
    elif fusion_type == "cross-encoder":# Using a cross-encoder to rerank the documents.
        # Combine and deduplicate the sparse and dense docs
        candidate_ids = set()
        for doc in sparse_docs + dense_docs:
            candidate_ids.add(doc["action_id"])
        candidate_docs_parsed = [get_parsed_action_by_id(id=action_id, context=context) for action_id in candidate_ids]

        # Get the cross-encoder scores for the candidates
        scores = cross_encoder_scores(query_string, candidate_docs_parsed)
        # Get the top k+offset scores ordered
        top_scores = torch.topk(torch.tensor(scores), k=k+offset)

        results = []
        for rank, idx in enumerate(top_scores.indices):
            if rank < offset:
                continue
            result = get_parsed_action_metadata(action=candidate_docs_parsed[idx], context=context)
            result["rank"] = rank + 1
            results.append(result)
    
        return results
    else:
        logging.error("Invalid fusion type given to hybrid document retriever in utils/action_retrieval.hybrid_retrieve_docs")
        return []


def main():
    logging.basicConfig(filename="logfiles/rag.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    context = ActionParsingContext(
        required_fields=["action_id", "action_title", "key_messages"]
    )
    docs = hybrid_retrieve_docs(
        query_string="timeframe prohibiting all fishing in a marine protected area leads to increased fish abundance",
        context=context,
        k=3,
        offset=3
    )
    print(docs)


if __name__ == "__main__":
    main()