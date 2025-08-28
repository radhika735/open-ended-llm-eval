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


class ActionRetrievalContext():
    def __init__(self, required_fields = ["action_id", "action_title", "key_messages"]):
        # doc_type can be "km" or "bg_km", with "km" for key messages and "bg_km" for background key messages
        self.__set_required_fields(required_fields=required_fields)
        self.__set_metadata_fields(required_fields=required_fields)
        self.__set_doc_type(required_fields=required_fields)
    
    def __set_required_fields(self, required_fields):
        # remove all instances of action_id and action_title from the list and self prepend for consistency:
            # self.__required_fields must begin with the elements "action_id" then "action_title" and there must not be duplicates of these in the required fields.
        # filtering out any elements which are "action_id" or "action_title":
        self.__required_fields = [f for f in required_fields if f not in ["action_id", "action_title"]]
        self.__required_fields = ["action_id", "action_title"] + self.__required_fields

    def __set_metadata_fields(self, required_fields):
        self.__metadata_fields = [f for f in required_fields if f not in ["key_messages", "background_information", "supporting_evidence"]]

    def __set_doc_type(self, required_fields):
        if "key_messages" in required_fields:
            if "background_information" in required_fields:
                self.__doc_type = "bg_km"
            else:
                self.__doc_type = "km"

    def get_required_fields(self):
        return self.__required_fields
    
    def get_metadata_fields(self):
        return self.__metadata_fields

    def get_doc_type(self):
        return self.__doc_type



def parse_action(action_string, context : ActionRetrievalContext):
    """
    Parse an action string into its components.
    
    Args:
        action_string (str): The action string to parse.
        context (ActionRetrievalContext): The context for action retrieval (i.e. can use this to find the user's set required_fields)

    Returns:
        dict: A dictionary containing the parsed action components.
    """
    required_fields = context.get_required_fields()
    all_action_fields = {}
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
        all_action_fields["background_information"] = background_information.strip()

    # Extract key messages.
    key_messages = "\n".join(lines[km_index:]) if km_index is not None else ""

    # Store extracted information.
        # (apart from background information which will optionally have been stored earlier)
    all_action_fields.update({
        "action_id": action_id.strip(),
        "action_title": action_title.strip(),
        "effectiveness": effectiveness.strip(),
        "key_messages": key_messages.strip()
    })

    # Only return the required fields.
    for field in required_fields:
        if field not in all_action_fields:
            logging.warning(f"Invalid required field '{field}' given to function parse_action")
        else:
            parsed_action[field] = all_action_fields[field]

    return parsed_action



def get_parsed_actions(context : ActionRetrievalContext):
    """
    Get parsed actions from the data directory.

    Args:
        context (ActionRetrievalContext): The context for action retrieval (e.g can use this to find the user's set doc_type, required_fields, metadata_fields)
    
    Returns:
        list: List of parsed action dictionaries
    """
    doc_type = context.get_doc_type()
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
                parsed_action = parse_action(action_string=file_contents, context=context)
                parsed_actions.append(parsed_action)

    return parsed_actions



def get_parsed_action_as_str(action):
    action_string = f"{action['action_id']}: {action['action_title']}"
    for k,v in action.items():
        if k not in ["action_id", "action_title"]:
            cleaned_key_name = " ".join(k.split("_"))
            cleaned_key_name = cleaned_key_name.title()
            action_string += f"\n{cleaned_key_name}: {v}"
    return action_string



def get_parsed_action_metadata(action, context : ActionRetrievalContext):
    metadata_fields = context.get_metadata_fields()
    metadata = {}
    for k,v in action.items():
        if k in metadata_fields:
            metadata[k] = v
    return metadata



def sparse_retrieve_docs(query_string, context : ActionRetrievalContext, k=3, offset=0):
    """
    Perform a sparse retrieval of documents based on a query string.
    
    Args:
        query_string (str): The search query for finding relevant actions
        context (ActionRetrievalContext): The context for action retrieval (e.g can use this to find the user's set doc_type, required_fields, metadata_fields)
        k (int): Number of top results to return (default: 3)
        offset (int): Number of results to skip (default: 0, used for pagination)
    
    Returns:
        list: Top k action documents matching the query, starting from offset
    """
    parsed_actions = get_parsed_actions(context=context)
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



def get_dense_embeddings(all_docs, context : ActionRetrievalContext, load_from_cache=True, save_to_cache=True, np_cache_dir="answer_gen_data/retrieval_methods/dense_embeddings_cached"):
    doc_type = context.get_doc_type()
    np_cache_file = os.path.join(np_cache_dir, f"{doc_type}_nomic_dense_embeddings.npy")
    if load_from_cache and os.path.exists(np_cache_file):
        return np.load(np_cache_file)
    else:
        import create_dense_embeddings
        embeddings = create_dense_embeddings.get_embeddings(docs=all_docs, save_to_cache=save_to_cache, cache_file=np_cache_file) # will take half an hour to run
        return embeddings



def dense_retrieve_docs(query_string, context : ActionRetrievalContext, k=3, offset=0):
    """
    Perform a dense retrieval of documents based on a query string.

    Args:
        query_string (str): The search query for finding relevant actions
        context (ActionRetrievalContext): The context for action retrieval (e.g can use this to find the user's set doc_type, required_fields, metadata_fields)
        k (int): Number of top results to return (default: 3)
        offset (int): Number of results to skip (default: 0, used for pagination)

    Returns:
        list: Top k action documents matching the query, starting from offset
    """
    parsed_actions = get_parsed_actions(context=context)
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
def hybrid_retrieve_docs(query_string, context : ActionRetrievalContext, k=3, offset=0):
    sparse_docs = sparse_retrieve_docs(query_string=query_string, context=context, k=k, offset=offset)
    dense_docs = dense_retrieve_docs(query_string=query_string, context=context, k=k, offset=offset)

    # # Combine and rank the results
    # combined_results = reciprocal_rank_fusion([sparse_docs, dense_docs], num_docs=k)
    # return combined_results

    # Using a cross-encoder:
    candidate_docs = list(set(sparse_docs).union(set(dense_docs)))
    scores = cross_encoder_scores(query_string, candidate_docs)
    flattened_scores = [score[0] for score in scores]

    if len(scores) >= 2:
        top_docs = [candidate_docs[i] for i in np.argsort(flattened_scores)[-2:][::-1]]
    else:
        top_docs = candidate_docs[:len(flattened_scores)]

    return top_docs



def main():
    logging.basicConfig(filename="logfiles/action_retrieval.log", level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    context = ActionRetrievalContext(
        required_fields=["action_id", "action_title", "key_messages"]
    )

    ## Testing parsed actions with bg km
    docs = get_parsed_actions(context=context)
    for i in range(100, 105):
        print(docs[i])




if __name__ == "__main__":
    main()