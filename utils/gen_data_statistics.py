from collections import Counter
from collections import defaultdict
import json
import logging
import os
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from utils.create_dense_embeddings_general import get_embeddings


def get_qus_for_synopsis(synopsis, qus_dir, action_doc_type="bg_km"):
    no_gaps_synopsis = "".join(synopsis.split())
    qus_filepath = f"{qus_dir}/{action_doc_type}_{no_gaps_synopsis}_qus.json"
    try:
        with open(qus_filepath, 'r', encoding="utf-8") as file:
            qus_list = json.load(file)
        return qus_list
    except FileNotFoundError:
        logging.error(f"Questions file {qus_filepath} not found.")
        return []
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {qus_filepath}.")
        return []


def get_n_representative_qus(questions, qu_embeddings=None, n=10, embedding_model_name="nomic-ai/nomic-embed-text-v1.5"):
    # Can either take either precomputed embeddings for the questions or not - if not, the embeddings are generated from the question using the model embedding_model_name. 
    if len(questions) <= n:
        return questions
    else:
        if (qu_embeddings is None) or (len(qu_embeddings) != len(questions)):
            all_embeddings = get_embeddings(text=questions, model_name=embedding_model_name)
        else:
            all_embeddings = qu_embeddings
    kmeans = KMeans(n_clusters=n, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(all_embeddings)
    representatives = []
    for i in range(n):
        cluster_indices = np.where(labels == i)[0]
        cluster_embeddings = all_embeddings[cluster_indices]
        centroid = kmeans.cluster_centers_[i]
        # Find the closest real question to the centroid.
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closest_index = cluster_indices[np.argmin(distances)]
        representatives.append(closest_index)
    top_qus = [questions[i] for i in representatives]
    return top_qus
    

def get_qu_distances_from_species(qus_list, species, qu_embeddings=None, species_embedding=None, embedding_model_name="nomic-ai/nomic-embed-text-v1.5"):
    # Can either take precomputed embeddings for the questions/species or not - if not, the embeddings are generated from the questions/species using the model embedding_model_name. 
    if (not qus_list) or (not species):
        return []
    else:
        if qu_embeddings is None:
            qu_embeddings = get_embeddings(text=qus_list, model_name=embedding_model_name)
        if species_embedding is None:
            species_embedding = get_embeddings(text=species, model_name=embedding_model_name)
            
    distances = []
    for e in qu_embeddings:
        distance = cosine_similarity(species_embedding.reshape(1,-1), e.reshape(1,-1))
        distances.append(distance)
    return distances


def get_n_representative_qus_for_synopsis(qus_list, synopsis, n=10, embedding_model_name="nomic-ai/nomic-embed-text-v1.5"):
    synopsis_words = synopsis.split()
    generic_words = ["conservation", "control", "management", "of", "sustainable"]
    species_words = [w for w in synopsis_words if w.lower() not in generic_words]
    species = " ".join(species_words)

    qu_embeddings = get_embeddings(text=qus_list, model_name=embedding_model_name)
    distances = get_qu_distances_from_species(qus_list=qus_list, qu_embeddings=qu_embeddings, species=species, embedding_model_name=embedding_model_name)
    relevant_qus = []
    relevant_qu_embeddings = []
    for q,e,d in zip(qus_list, qu_embeddings, distances):
        if d >= 0.5:
            relevant_qus.append(q)
            relevant_qu_embeddings.append(e)
        else:
            print("Irrelevant question:",q)
    relevant_qu_embeddings = np.array(relevant_qu_embeddings)
    top_qus = get_n_representative_qus(questions=relevant_qus, qu_embeddings=relevant_qu_embeddings, n=n, embedding_model_name=embedding_model_name)
    return top_qus


def get_id_dist_for_synopsis(synopsis, qus_dir, action_doc_type="bg_km"):
    qus_list = get_qus_for_synopsis(synopsis=synopsis, qus_dir=qus_dir, action_doc_type=action_doc_type)
    used_ids_list = []
    for qu_dict in qus_list:
        used_ids_list.extend(qu_dict["all_relevant_action_ids"])
    id_counts = Counter(used_ids_list)
    return id_counts


def get_id_dist_all_synopses(qus_dir, action_doc_type="bg_km"):
    full_id_dist = {}
    for entry in os.scandir("action_data/key_messages/km_synopsis"):
        synopsis = entry.name
        synopsis_id_dist = get_id_dist_for_synopsis(synopsis=synopsis, qus_dir=qus_dir, action_doc_type=action_doc_type)
        full_id_dist[synopsis] = synopsis_id_dist
    return full_id_dist


def get_num_ids_used_for_synopsis(synopsis, qus_dir, action_doc_type="bg_km"):
    id_dist = get_id_dist_for_synopsis(synopsis=synopsis, qus_dir=qus_dir, action_doc_type=action_doc_type)
    return len(id_dist)


def get_num_qus_for_synopsis(synopsis, qus_dir, action_doc_type="bg_km"):
    qus_list = get_qus_for_synopsis(synopsis=synopsis, qus_dir=qus_dir, action_doc_type=action_doc_type)
    return len(qus_list)


def get_num_qus_all_synopses(qus_dir, action_doc_type="bg_km"):
    num_qus_by_synopsis = {}
    for entry in os.scandir("action_data/key_messages/km_synopsis"):
        synopsis = entry.name
        num_qus = get_num_qus_for_synopsis(synopsis=synopsis, qus_dir=qus_dir, action_doc_type=action_doc_type)
        num_qus_by_synopsis[synopsis] = num_qus
    return num_qus_by_synopsis


def get_total_num_qus(qus_dir, action_doc_type="bg_km"):
    num_qus_by_synopsis = get_num_qus_all_synopses(qus_dir=qus_dir, action_doc_type=action_doc_type)
    total = sum(num_qus_by_synopsis.values())
    return total


def print_ids_used_per_synopsis(id_dist):
    for [synopsis,ids] in id_dist:
        print(synopsis,ids)


def write_distribution_file(id_dist, filepath):
    with open(filepath, 'w') as file:
        for [synopsis, ids] in id_dist:
            file.write(synopsis + "," + ids + "\n")


def parse_model_name(model):
    model_split = model.split("/")
    model_name = model_split[-1]
    cleaned_name = ""
    for char in model_name:
        if char.isalnum():
            cleaned_name += char
        else:
            cleaned_name += "-"
    return cleaned_name


def parse_provider_name(provider):
    if provider is not None:
        provider_split = provider.split("/")
        provider_name = provider_split[0]
        return provider_name
    else:
        return ""


def get_summary_gen_qus_usage_separate(model, provider, base_qu_dir="live_questions", qu_types=["answerable", "unanswerable"], filter_stages=["passed", "failed"]):
    usage = []

    for qu_type in qu_types:
        for stage in filter_stages:
            dir = f"{base_qu_dir}/bg_km_qus/{qu_type}/{stage}/usage_annotated"
            num_qus_used = 0
            num_qus_unused = 0
            for filename in os.listdir(dir):
                filepath = os.path.join(dir, filename)
                with open(filepath, 'r', encoding="utf-8") as f:
                    qus = json.load(f)
                for qu in qus:
                    if [model, provider] in qu.get("used_by_models", []):
                        num_qus_used += 1
                    else:
                        num_qus_unused += 1
            usage.append({
                "qu_type": qu_type,
                "filter_stage": stage,
                "used": num_qus_used,
                "unused": num_qus_unused
            })

    return usage


def get_summary_gen_qus_usage_combined(model, provider, base_qu_dir="live_questions", qu_types=["answerable", "unanswerable"], filter_stages=["passed", "failed"]):
    total_used = 0
    total_unused = 0

    usage = get_summary_gen_qus_usage_separate(model=model, provider=provider, base_qu_dir=base_qu_dir, qu_types=qu_types, filter_stages=filter_stages)
    for u in usage:
        total_used += u["used"]
        total_unused += u["unused"]

    return {
        "used": total_used,
        "unused": total_unused
    }


def get_viable_summaries_split_for_model(model_provider):
    num_viable = 0
    num_non_viable = 0
    print(model_provider)
    summaries_dir = os.path.join("summary_gen_data/answerable_passed_qus_summaries/hybrid_cross-encoder", model_provider)
    print(summaries_dir)
    for summaries_file in sorted(os.listdir(summaries_dir)):
        if summaries_file.endswith(".json"):
            summaries_filepath = os.path.join(summaries_dir, summaries_file)
            with open(summaries_filepath, 'r', encoding="utf-8") as f:
                summaries = json.load(f)
            for summary in summaries:
                if summary["relevant_summary"] is None:
                    num_non_viable += 1
                else:
                    num_viable += 1
    
    return {
        "num_viable": num_viable,
        "num_non_viable": num_non_viable
    }


def print_viable_summaries_split(joined_model_provider_list):
    for model_provider in joined_model_provider_list:
        viable_summaries_split = get_viable_summaries_split_for_model(model_provider=model_provider)
        print(f"{model_provider}: {viable_summaries_split}")


def get_evals_generated_separate(
        judge_model, 
        judge_provider, 
        base_summaries_dir="live_summaries", 
        qu_types=["answerable", "unanswerable"], 
        filter_stages=["passed", "failed"], 
        retrieval_types=["hybrid_cross-encoder"],
        answering_models_providers_list=[
            ("openai/gpt-5", None),
            ("anthropic/claude-sonnet-4", None),
            ("google/gemini-2.5-pro", None),
            ("moonshotai/kimi-k2-0905", "fireworks/fp8")
        ]
    ):
    eval_nums = []

    for qu_type in qu_types:
        for stage in filter_stages:
            for retrieval_type in retrieval_types:
                for answer_model, answer_provider in answering_models_providers_list:
                    answer_model_cleaned = parse_model_name(answer_model)
                    answer_provider_cleaned = parse_provider_name(answer_provider)
                    answer_mp_cleaned = f"{answer_provider_cleaned}_{answer_model_cleaned}"
                    dir = f"{base_summaries_dir}/{qu_type}_{stage}_qus_summaries/{retrieval_type}/eval_annotated/{answer_mp_cleaned}"
                    num_summaries_evaluated = 0
                    num_summaries_unevaluated = 0
                    for filename in os.listdir(dir):
                        filepath = os.path.join(dir, filename)
                        with open(filepath, 'r', encoding="utf-8") as f:
                            summaries = json.load(f)
                        for summary in summaries:
                            summary_evaluated = False
                            for judge in summary.get("eval_by_models", []):
                                if judge["judge_model"] == judge_model and judge["judge_provider"] == judge_provider:
                                    summary_evaluated = True
                            if summary_evaluated:
                                num_summaries_evaluated += 1
                            else:
                                num_summaries_unevaluated +=1
                                    
                    eval_nums.append({
                        "qu_type": qu_type,
                        "filter_stage": stage,
                        "retrieval_type": retrieval_type,
                        "answering_model_provider": answer_mp_cleaned,
                        "num_summaries_evaluated": num_summaries_evaluated,
                        "num_summaries_unevaluated": num_summaries_unevaluated
                    })

    return eval_nums


def get_evals_generated_combined(
        judge_model, 
        judge_provider, 
        base_summaries_dir="live_summaries", 
        qu_types=["answerable", "unanswerable"], 
        filter_stages=["passed", "failed"], 
        retrieval_types=["hybrid_cross-encoder"],
        answering_models_providers_list=[
            ("openai/gpt-5", None),
            ("anthropic/claude-sonnet-4", None),
            ("google/gemini-2.5-pro", None),
            ("moonshotai/kimi-k2-0905", "fireworks/fp8")
        ]
    ):
    num_summaries_evaluated = 0
    num_summaries_unevaluated = 0

    evals_generated = get_evals_generated_separate(
        judge_model=judge_model, 
        judge_provider=judge_provider, 
        base_summaries_dir=base_summaries_dir, 
        qu_types=qu_types, 
        filter_stages=filter_stages,
        retrieval_types=retrieval_types,
        answering_models_providers_list=answering_models_providers_list,
    )
    for e in evals_generated:
        num_summaries_evaluated += e["num_summaries_evaluated"]
        num_summaries_unevaluated += e["num_summaries_unevaluated"]

    return {
        "num_summaries_evaluated": num_summaries_evaluated,
        "num_summaries_unevaluated": num_summaries_unevaluated
    }
    


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    #num_qus = get_data(type="num_qus")
    #write_distribution_file(id_dist=id_dist, filepath="question_gen_data/km_multi_action_data/action_id_dist_to_rm.txt")
    #print_ids_used_per_synopsis(id_dist)
    #print(f"TOTAL NUM QUS: {num_qus}")
    # synopsis = "Natural Pest Control"
    # no_gaps_synopsis = "".join(synopsis.split())
    # qus_file = f"question_gen_data/bg_km_multi_action_data/bg_km_multi_action_gen_qus/answerable/bg_km_{no_gaps_synopsis}_qus.json"
    # id_dist = get_id_dist_for_synopsis(qus_file=qus_file, synopsis=synopsis)
    # print(id_dist)
    # qus_dir="question_gen_data/bg_km_multi_action_data/bg_km_qus/answerable/all"
    # print(get_num_qus_all_synopses(qus_dir=qus_dir, action_doc_type="bg_km"))
    # synopsis = "Bat Conservation"
    # synopsis_qus = get_qus_for_synopsis(synopsis=synopsis, qus_dir=qus_dir, action_doc_type="bg_km")
    # synopsis_queries = [qu_dict["question"] for qu_dict in synopsis_qus]
    # top_qus = get_top_n_common_qus(questions=synopsis_queries, n=15)

    # distances = get_qu_distances_from_species(qus_list=synopsis_queries, species="Bat")
    # for q,d in zip(synopsis_queries, distances):
    #     print(q,d)

    # top_qus = get_n_representative_qus_for_synopsis(qus_list=synopsis_queries, synopsis=synopsis)
    # for q in top_qus:
    #     print(f"{q}")

    # total = get_total_num_qus(qus_dir=qus_dir)
    # print(total)
    answering_model_provider_list = [
        ("openai/gpt-5", None),
        ("anthropic/claude-sonnet-4", None),
        ("google/gemini-2.5-pro", None),
        ("moonshotai/kimi-k2-0905", "fireworks/fp8")
    ]

    ### SEE THE NUMBER OF GENERATED SUMMARIES WHICH ARE USABLE AND WHICH ARE NULL
    # cleaned_mp_list = [f"{parse_provider_name(p)}_{parse_model_name(m)}" for m,p in model_provider_list]
    # for mp in cleaned_mp_list:
    #     viable_summaries_split = get_viable_summaries_split_for_model(model_provider=mp)
    #     print(f"{mp}: {viable_summaries_split}")

    ### SEE THE NUMBER OF SUMMARIES GENERATED
    for model, provider in answering_model_provider_list:
        usage = get_summary_gen_qus_usage_separate(model, provider, base_qu_dir="live_questions", qu_types=["answerable", "unanswerable"], filter_stages=["passed", "failed"])
        print(f"\n\nModel: {model}, Provider: {provider}, Usage: {usage}")

    ### SEE THE NUMBER OF EVALS GENERATED
    judge_model_provider_list = [
        ("google/gemini-2.5-pro", None),
        ("google/gemini-2.5-flash", None),
        ("openai/gpt-5", None),
        ("openai/gpt-5-mini", None)
    ]
    for m, p in judge_model_provider_list:
        evals_generated = get_evals_generated_combined(judge_model=m, judge_provider=p)#, base_summaries_dir="live_summaries", qu_types=["answerable", "unanswerable"], filter_stages=["passed", "failed"], retrieval_types=["hybrid_cross-encoder"], answering_models_providers_list=answering_model_provider_list)
        print(f"\n\nJudge Model: {m}, Judge Provider: {p}, Evals Generated: {evals_generated}")


if __name__=="__main__":
    main()