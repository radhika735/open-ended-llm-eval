from collections import defaultdict
import os
import re
import shutil
source = 'action_data/background_key_messages/bg_km_all'

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    
def main():
    d = defaultdict(int)
    for filename in os.listdir(source):
        file_path = os.path.join(source, filename)
        content = read_file(file_path)
        
        synopsis = (content.partition("Title: ")[2].partition("\nDescription: ")[0])
        target = f"action_data/background_key_messages/bg_km_synopsis/{synopsis}"

        os.makedirs(target, exist_ok=True)
        shutil.copyfile(file_path, os.path.join(target, filename))

        d[synopsis] += 1


    print(d)


if __name__ == "__main__":
    main()