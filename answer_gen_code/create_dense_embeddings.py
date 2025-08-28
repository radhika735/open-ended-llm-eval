from itertools import batched
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


def work_init():
    global _MODEL
    from sentence_transformers import SentenceTransformer
    _MODEL = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)


def embed_chunk(doc_info):
    index, doc_chunks = doc_info
    print("Embedding docs from Document", doc_chunks[0][:50], flush=True)
    global _MODEL
    embeddings = _MODEL.encode(doc_chunks, batch_size=32, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    return index, embeddings


def get_embeddings(docs, save_to_cache=True, cache_file="answer_gen_data/retrieval_methods/dense_embeddings_cached/unknowndoctype_nomic_dense_embeddings.npy"):
    batch_size = 400
    batches = [list(b) for b in batched(docs, batch_size)]

    print("about to start")
    results = []
    done = 0
    start = time.monotonic()

    with ProcessPoolExecutor(max_workers=4, initializer=work_init) as executor:
        future_to_batch = {
            executor.submit(embed_chunk, (i,batch)): i for i, batch in enumerate(batches)
        }

        for future in as_completed(future_to_batch):
            try:
                batch_idx, embeddings = future.result()
                results.append((batch_idx, embeddings))
                done += 1
                print(f"Done {done}/{len(batches)}. Just did batch {batch_idx}")
                print(f"Elapsed time: {time.monotonic() - start}")
                print("\n")

            except Exception as exc:
                batch_idx = future_to_batch[future]
                print(f"Batch {batch_idx} generated an exception: {exc}", flush=True)
    

    results = sorted(results)
    flat_results = np.vstack([embeddings for _, embeddings in results])
    if save_to_cache:
        np.save(cache_file, flat_results)
    return flat_results

    
    
    

    