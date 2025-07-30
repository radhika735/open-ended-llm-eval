from collections import Counter
import json
import os

def get_data(type="id_dist"):
    # type options: "num_qus", "id_dist", "id_rm_dist"
    id_dist = []
    id_rm_dist = []
    total_num_qus = 0

    for entry in os.scandir("action_data/key_messages/km_synopsis"):
        synopsis = entry.name
        no_gaps_synopsis = "".join(synopsis.split())
        QUS_FILE = f"exam_gen_data/multi_action_data/km_{no_gaps_synopsis}_qus.json"

        with open(QUS_FILE, 'r', encoding='utf-8') as file:
            qus_list = json.load(file)

        action_id_list = []

        for qu_obj in qus_list:
            action_id_list.extend(qu_obj["action_ids_used_for_question_generation"])

        id_counts = Counter()

        for id in action_id_list:
            id_counts[id] += 1

        id_list = list(id_counts.keys())
        id_rm_list = [id for id in id_list if id_counts[id] >= 2]
        
        id_list_str = ""
        for id in id_list:
            id_list_str += f"{id} "
        id_list_str.rstrip()

        id_rm_list_str = ""
        for id in id_rm_list:
            id_rm_list_str += f"{id} "
        id_rm_list_str = id_rm_list_str.rstrip()

        id_rm_dist.append([synopsis,id_rm_list_str])
        id_dist.append([synopsis,id_list_str])
        total_num_qus += len(qus_list)
        print(f"SYNOPSIS: {synopsis}")
        print(f"NUMBER OF QUESTIONS: {len(qus_list)}")
        print(f"DISTRIBUTION: {id_counts}")
        print(f"NUM ACTION IDS: {len(id_counts)}")
        print(f"ACTION ID LIST: {id_list_str}")
        print(f"ACTION IDS TO REMOVE LIST: {id_rm_list_str}")
        print("\n")

    if type=="id_dist":
        return id_dist
    elif type=="id_rm_dist":
        return id_rm_dist
    elif type=="num_qus":
        return total_num_qus


def print_ids_used_per_synopsis(id_dist):
    for [synopsis,ids] in id_dist:
        print(synopsis,ids)


def write_distribution_file(id_dist, filepath):
    with open(filepath, 'w') as file:
        for [synopsis, ids] in id_dist:
            file.write(synopsis + "," + ids + "\n")


def main():
    num_qus = get_data(type="num_qus")
    #write_distribution_file(id_dist=id_dist, filepath="exam_gen_data/action_id_dist_to_rm.txt")
    #print_ids_used_per_synopsis(id_dist)
    print(f"TOTAL NUM QUS: {num_qus}")

if __name__=="__main__":
    main()