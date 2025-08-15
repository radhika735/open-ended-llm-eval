import os
import Stemmer
import logging
import bm25s
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict


def parse_action(action_string):
    """
    Parse an action string into its components.
    
    Args:
        action_string (str): The action string to parse.
    
    Returns:
        dict: A dictionary containing the parsed action components.
    """
    parsed_action = {}
    lines = action_string.strip().splitlines()

    # Remove the line "Synopsis Details:" and lines after it.
    for i, line in enumerate(lines):
        if line == "Synopsis Details:":
            lines = lines[:i]
            break

    # Extract action id, action title and effectiveness rating.
    action_id, action_title = lines[0].split(": ", 1)
    effectiveness = lines[1] if len(lines) > 1 else ""

    # Parse (optional) background information and (mandatory) key messages.
        # (whether background information is in the file depends on the contents of the action_string passed as argument).
    bg_index = None
    km_index = None

    for line in lines[2:]:
        if line.startswith("Background information and definitions:"):
            bg_index = lines.index(line)
        if line.startswith("Key Messages:"):
            km_index = lines.index(line)

    if bg_index is not None:
        # Background information exists in the action file, extract it and store it.
        bg_lines = lines[bg_index:km_index]
        background_information = "\n".join(lines[bg_index:km_index])
        parsed_action["background_information"] = background_information.strip()

    # Extract key messages.
    key_messages = "\n".join(lines[km_index:]) if km_index is not None else ""

    # Store extracted information.
        # (apart from background information which will optionally have been stored earlier)
    parsed_action.update({
        "action_id": action_id.strip(),
        "action_title": action_title.strip(),
        "effectiveness": effectiveness.strip(),
        "key_messages": key_messages.strip()
    })

    return parsed_action



def get_parsed_actions(doc_type = "km"):
    # doc_type can be "km" or "bg_km", with "km" for key messages and "bg_km" for background key messages
    """
    Get parsed actions from the data directory.
    
    Returns:
        list: List of parsed action dictionaries
    """
    parsed_actions = []
    
    if doc_type == "km":
        data_dir="action_data/key_messages/km_all"
    elif doc_type == "bg_km":
        data_dir="action_data/background_key_messages/bg_km_all"
    else:
        raise ValueError("Invalid doc_type. Use 'km' for key messages or 'bg_km' for background key messages.")
    
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".txt"):
            with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as action_file:
                file_contents = action_file.read()
                parsed_action = parse_action(file_contents)
                parsed_actions.append(parsed_action)


    return parsed_actions



def sparse_retrieve_docs(query_string, k=3, offset=0):
    """
    Perform a sparse retrieval of documents based on a query string.
    
    Args:
        query_string (str): The search query for finding relevant actions
        k (int): Number of top results to return (default: 3)
        offset (int): Number of results to skip (default: 0, used for pagination)
    
    Returns:
        list: Top k action documents matching the query, starting from offset
    """
    parsed_actions = get_parsed_actions(doc_type="km")
    corpus = []
    for action in parsed_actions:
        corpus.append(f"{action['action_id']}: {action['action_title']}\nEffectiveness: {action['effectiveness']}\nKey Messages:\n{action['key_messages']}")

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
        results.append({
            "action_id": action["action_id"],
            "action_title": action["action_title"],
            "effectiveness": action["effectiveness"],
            "rank": i + 1  # Keep original rank from search
        })
    
    logging.debug(f"Found {len(results)} documents for query '{query_string}':")
    return results



def get_dense_embeddings(all_docs, dense_model, doc_type = "km", load_from_cache=True, save_to_cache=True, np_cache_dir="answer_gen_data/retrieval_methods/dense_embeddings_cached"):
    np_cache_file = os.path.join(np_cache_dir, f"{doc_type}_nomic_dense_embeddings.npy")
    if load_from_cache and os.path.exists(np_cache_file):
        return np.load(np_cache_file)
    else:
        import create_dense_embeddings
        embeddings = create_dense_embeddings.get_embeddings(docs=all_docs, save_to_cache=save_to_cache) # will take half an hour to run
        return embeddings



def dense_retrieve_docs(query_string, k=3, offset=0):
    """
    Perform a dense retrieval of documents based on a query string.

    Args:
        query_string (str): The search query for finding relevant actions
        k (int): Number of top results to return (default: 3)
        offset (int): Number of results to skip (default: 0, used for pagination)

    Returns:
        list: Top k action documents matching the query, starting from offset
    """
    parsed_actions = get_parsed_actions(doc_type="bg_km")
    corpus = []
    for action in parsed_actions:
        corpus.append(f"{action['action_id']}: {action['action_title']}\nEffectiveness: {action['effectiveness']}\nBackground Information:\n{action['background_information']}\nKey Messages:\n{action['key_messages']}")

    # Encode the query
    model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    query_embedding = model.encode(query_string, normalize_embeddings=True)
    # Get the document embeddings
    dense_embeddings = get_dense_embeddings(all_docs=corpus, model=model, np_cache_file="answer_gen_data/retrieval_methods/dense_embeddings_cached/bg_km_nomic_dense_embeddings.npy")

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
        results.append({
            "action_id": action["action_id"],
            "action_title": action["action_title"],
            "effectiveness": action["effectiveness"],
            "rank": rank + 1
        })
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



def hybrid_retrieve_docs(query_string, k=3, offset=0):
    sparse_docs = sparse_retrieve_docs(query_string=query_string, k=k, offset=offset)
    dense_docs = dense_retrieve_docs(query_string=query_string, k=k, offset=offset)

    # Combine and rank the results
    combined_results = reciprocal_rank_fusion([sparse_docs, dense_docs], num_docs=k)
    return combined_results



def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    ## Testing parsed actions with bg km
    docs = get_parsed_actions(doc_type="bg_km")
    for i in range(100, 105):
        print(docs[i])



if __name__ == "__main__":
    main()