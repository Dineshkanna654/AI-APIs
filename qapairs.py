import csv
import json
import spacy

nlp = spacy.load("en_core_web_sm")

def generate_qa_pairs(csv_file):
    qa_pairs = []
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            customer_name = row['customer_name']
            total_value = str(calculate_total_value(csv_file, customer_name))
            qa_pairs.append({
                "question": f"What is the Total value or Net value for {customer_name}?",
                "answer": total_value
            })
            qa_pairs.append({
                "question": f"What is the product code for {row['product_description']}?",
                "answer": row['product_code']
            })
            qa_pairs.append({
                "question": f"Who is the customer for the product {row['product_description']}?",
                "answer": row['customer_name']
            })
            qa_pairs.append({
                "question": f"Who is the customer for the product code {row['product_code']}?",
                "answer": row['customer_name']
            })
            qa_pairs.append({
                "question": f"What is the product description for this product code {row['product_code']}?",
                "answer": row['product_description']
            })
            qa_pairs.append({
                "question": f"What is the total value for this product code {row['product_code']}?",
                "answer": row['total_value']
            })
            qa_pairs.append({
                "question": f"What is the invoice number for this product code {row['product_code']}?",
                "answer": row['invoice_no']
            })
            qa_pairs.append({
                "question": f"What is the invoice date for this product code {row['product_code']}?",
                "answer": row['invoice_date']
            })
            qa_pairs.append({
                "question": f"What is the ordered quantity for this product code {row['product_code']}?",
                "answer": row['order_qty']
            })
            qa_pairs.append({
                "question": f"What is the MRP for this product code {row['product_code']}?",
                "answer": row['item_mrp']
            })
            qa_pairs.append({
                "question": f"Who is the customer for this invoice number {row['invoice_no']}?",
                "answer": row['customer_name']
            })
            qa_pairs.append({
                "question": f"What is the date of for this invoice number {row['invoice_no']}?",
                "answer": row['invoice_date']
            })
            qa_pairs.append({
                "question": f"When was the invoice for {row['product_description']} issued to {row['customer_name']}?",
                "answer": row['invoice_date']
            })
            qa_pairs.append({
                "question": f"What is the total value of the order for {row['product_description']} for {row['customer_name']}?",
                "answer": row['total_value']
            })
            qa_pairs.append({
                "question": f"How many items were ordered for the product {row['product_description']} by {row['customer_name']}?",
                "answer": row['order_qty']
            })
            qa_pairs.append({
                "question": f"What is the invoice number for the product {row['product_description']} ordered by {row['customer_name']}?",
                "answer": row['invoice_no']
            })
            qa_pairs.append({
                "question": f"What is the item MRP for the product {row['product_description']}?",
                "answer": row['item_mrp']
            })
    return qa_pairs

def get_column_values(csv_file, column_name):
    column_values = []

    with open(csv_file, 'r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            column_values.append(row[column_name])

    return column_values

def calculate_total_value(csv_file, customer_name):
    total_value = 0
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['customer_name'] == customer_name:
                total_value += float(row['total_value'])
    return total_value


def save_qa_pairs_as_json(qa_pairs, json_file):
    with open(json_file, 'w') as file:
        json.dump(qa_pairs, file, indent=4)

def dataset():
    csv_file = "erp100.csv"
    qa_pairs = generate_qa_pairs(csv_file)
    return qa_pairs

data = dataset()

def remove_duplicate_pairs(qa_pairs):
    unique_pairs = set()
    unique_qa_pairs = []

    for pair in qa_pairs:
        pair_tuple = (pair['question'], pair['answer'])
        if pair_tuple not in unique_pairs:
            unique_pairs.add(pair_tuple)
            unique_qa_pairs.append(pair)
    return unique_qa_pairs
 
cleaned_data = remove_duplicate_pairs(data)

def CleanedDataSet():
    data = dataset()
    cleaned_data = remove_duplicate_pairs(data)
    return cleaned_data

if __name__ == "__main__":
    csv_file = "erp100.csv"
    json_file = "qa_pairs.json"

    qa_pairs = generate_qa_pairs(csv_file)
    save_qa_pairs_as_json(cleaned_data, json_file)

# https://chat.openai.com/c/a28d39c4-c001-453c-9a89-c75af44c803e

