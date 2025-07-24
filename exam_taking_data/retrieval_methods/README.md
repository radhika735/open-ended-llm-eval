`retrieval_analysis` to get retrieval results based on a CSV file (in the format out from  `preprocessor.py` in `examtaking`). 
*Important* (have not fixed this): the CSV files will always cause errors (UnicodeCharErrors). Copying the data and saving it as a CSV UTF-8 delimited file usually works. (Easier than trying to find the errors - from CE database characters). 


#### Structure 

Multiple numpy files exist within the folder `retrievalmethods/dense_embeddings_cached`, where  embeddings have been cached:

- `km_alt_dense_embeddings.npy` is for `multi-qa-MiniLM-L6-cos-v1`.
- `km_minisix_dense_embeddings.npy` is for `all-MiniLM-L6-v2`.
- `km_nomic_dense_embeddings.npy` is for `nomic-ai/nomic-embed-text-v1.5`. This is what is currently being used. See Google Doc for comparison results. 
- `km_efive_dense_embeddings.npy` is for `intfloat/e5-large-v2`. 

The `cached_models` folder, in the same directory as the above, contains a cached Cross Encoder for hybrid search. 
    This is the same Cross Encoder used by the original research paper - `ms-marco-MiniLM-L-6-v2`. 

The CSV files are *appended to* not overwritten. If testing the same question multiple times, you need to manually delete irrelevant rows. 

