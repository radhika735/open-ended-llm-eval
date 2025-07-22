import os
import random
import shutil

root_dir = "preparation\key_messages\synopsis"
dest_dir = "preparation\key_messages\selected"

os.makedirs(dest_dir, exist_ok=True)

for dirpath, dirnames, filenames in os.walk(root_dir):
    if dirpath == root_dir or dirpath == dest_dir:
        continue
    
    if filenames:
        random_files = random.sample(filenames, 3) #All synopses have > 3 actions with evidence
        
        for random_file in random_files:
            src_path = os.path.join(dirpath, random_file)
            dst_path = os.path.join(dest_dir, f"{random_file}")
            
            shutil.copy2(src_path, dst_path)

