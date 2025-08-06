import csv

RETRIEVAL_MECH = "Survey_sparse"

def analyse_csv(file_path):
    matching_numbers = 0
    matching_synopses = 0
    first_numbers = 0
    total_rows = 0
    questions = set()

    with open(file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            total_rows += 1

            if row['Source No.'] == row['Retrieved No.']:
                matching_numbers += 1
                if row['Source No.'] not in questions:
                    first_numbers += 1
                    questions.add(row["Source No."])
            if row['Source Synopsis'] == row['Retrieved Synopsis']:
                matching_synopses += 1
            
            

    print(f"Total number of questions: {total_rows/2}")
    print(f"Total rows: {total_rows}")
    print(f"Number of times Source No. = Retrieved No.: {matching_numbers}")
    print(f"Percentage correctly retrieved doc was first {first_numbers /  matching_numbers * 100:.2f}")

    #Since we retrieve two distinct documents for each, and only one can be the correct doc, we double the percentage of direct hits. 
    print(f"Percentage of hits (will be one of 2 docs if success): {2 * (matching_numbers / total_rows) * 100:.2f}%")
    print(f"Number of times Source Synopsis = Retrieved Synopsis: {matching_synopses}")
    print(f"Percentage: {(matching_synopses / total_rows) * 100:.2f}%")

analyse_csv(f'retrievalmethods/{RETRIEVAL_MECH}_retrieval_info.csv')