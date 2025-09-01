def get_n_unique_qus(qu_dicts, n=10):
    n_unique_dicts = []
    remaining_qu_dicts = []
    for i, q_dict in enumerate(qu_dicts):
        if len(n_unique_dicts) >= n:
            break
        query = q_dict["question"]
        current_queries = [d["question"] for d in n_unique_dicts]
        if query in current_queries:
            remaining_qu_dicts.append(q_dict)
        else:
            n_unique_dicts.append(q_dict)
    unprocessed_dicts = qu_dicts[i:]
    remaining_qu_dicts.extend(unprocessed_dicts)
    return n_unique_dicts, remaining_qu_dicts


def get_unique_batches(qu_dicts, batch_size=10):
    unique_batches = []
    unprocessed_dicts = qu_dicts.copy()
    while unprocessed_dicts:
        new_batch, unprocessed_dicts = get_n_unique_qus(unprocessed_dicts, n=batch_size)
        unique_batches.append(new_batch)
    return unique_batches


l1 = [{'a':1,'b':2},{'a':3,'b':4},{'a':5,'b':6}]
l1copy = l1.copy()
l1copy[0]['a'] = 2
print(l1)
print(l1copy)