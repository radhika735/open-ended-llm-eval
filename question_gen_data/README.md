# Folder structure #

- km_synopsis_unfiltered_concat: 
    contains files containing all the action files (containing only key messages, but can be with or without supporting evidence) concatenated, for each synopsis.

- km_synopsis_filtered_concat:
    contains the same as above but with previously generated for action files removed (to avoid the LLM repeatedely generating responses using the same actions).
    km_synopsis_filtered_concat is created by running the exam_gen_code/multi_action_id_dist.py file to create the action_id_dist_config.txt file, and then running some bash scripts to generate the concatenatation of the filtered action files per synopsis.