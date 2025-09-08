from sentence_transformers import SentenceTransformer

def get_embeddings(text, model_name="nomic-ai/nomic-embed-text-v1.5"):
    model = SentenceTransformer(model_name, trust_remote_code=True)
    embeddings = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
    return embeddings