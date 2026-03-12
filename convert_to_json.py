import csv
import json

input_file = "faq_data.csv"
output_file = "faq.json"

faq_list = []

with open(input_file, mode="r", encoding="utf-8") as csv_file:
    reader = csv.DictReader(csv_file)
    
    for row in reader:
        faq = {
            "id": int(row["ID"]),
            "category": row["Category"],
            "question": row["Question"],
            "answer": row["Answer"]
        }
        faq_list.append(faq)

with open(output_file, mode="w", encoding="utf-8") as json_file:
    json.dump(faq_list, json_file, indent=4)

print("✅ faq.json file successfully created!")
