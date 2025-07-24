### UTILITY FILE - CAN GET RID OF THIS 
import os
import re
import shutil

source = "exam_gen_data/km_exam_gen_all_source"
evidence_target = "exam_gen_data/km_evid_source"
no_evidence_target = "exam_gen_data/km_noevid_source"


def main():
    for entry in os.scandir(source):
        try:
            with open(entry.path, 'r', encoding='utf-8') as file:
                content = file.read()

            if re.search("We found no studies", content, re.IGNORECASE | re.DOTALL) or re.search("We found no evidence", content, re.IGNORECASE | re.DOTALL) or re.search("we have not yet found any studies that directly and quantitatively tested this action", content, re.IGNORECASE | re.DOTALL):
                shutil.move(entry.path, os.path.join(no_evidence_target, entry.name))
            else:
                shutil.move(entry.path, os.path.join(evidence_target, entry.name))
        except Exception as e:
            print(f"Error processing {entry.name}: {e}")
            raise e



if __name__ == "__main__":
    main()