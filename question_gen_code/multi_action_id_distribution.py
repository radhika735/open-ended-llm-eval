from collections import Counter
from collections import defaultdict
import json
import logging
import os


def get_qus_for_synopsis(synopsis, qus_dir, action_doc_type="bg_km"):
    no_gaps_synopsis = "".join(synopsis.split())
    qus_filepath = f"{qus_dir}/{action_doc_type}_{no_gaps_synopsis}_qus.json"
    try:
        with open(qus_filepath, 'r', encoding="utf-8") as file:
            qus_list = json.load(file)
        return qus_list
    except FileNotFoundError:
        logging.error(f"Questions file {qus_filepath} not found.")
        exit()
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {qus_filepath}.")
        return []


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


def get_id_dist_to_rm(id_dist, cutoff=1):
    # cutoff number of times an id has to be used for it to be included in the id_dist for removal.
    return {synopsis: {id: count for id, count in id_counts.items() if count >= cutoff}
            for synopsis, id_counts in id_dist.items()}


def get_id_dist_as_str_list(id_dist):
    all_id_dist_str_list = []

    for synopsis, id_counts in id_dist.items():
        id_counts_str = ""
        for id in id_counts.keys():
            id_counts_str += f"{id} "
        id_counts_str = id_counts_str.rstrip()
        all_id_dist_str_list.append([synopsis, id_counts_str])

    return all_id_dist_str_list


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
    qus_dir="question_gen_data/bg_km_multi_action_data/bg_km_qus/answerable/all"
    print(get_num_qus_all_synopses(qus_dir=qus_dir, action_doc_type="bg_km"))


if __name__=="__main__":
    main()