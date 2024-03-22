from sklearn.metrics.pairwise import cosine_similarity
from qapairs import CleanedDataSet
import torch

def preprocess_data(data):
    questions = [item["question"] for item in data]
    answers = [item["answer"] for item in data]
    # Concatenate questions and answers for vectorization
    corpus = questions + answers
    return corpus, questions, answers

import re

def get_answer(query, vectorizer, tfidf_matrix, questions, answers):
    # Vectorize the query
    query_vector = vectorizer.transform([query])

    # Calculate cosine similarities with questions
    question_similarities = cosine_similarity(query_vector, tfidf_matrix[:len(questions)])

    # Calculate cosine similarities with answers
    answer_similarities = cosine_similarity(query_vector, tfidf_matrix[len(questions):])

    # Get index of the most similar question or answer
    question_index = question_similarities.argmax()
    answer_index = answer_similarities.argmax()

    # Check if the most similar is a question or answer
    if question_similarities[0, question_index] > answer_similarities[0, answer_index]:
        print("answers[question_index]", answers[question_index])
        return answers[question_index]
    else:
        # Check if the query contains 'total value' and a product code
        match = re.search(r'total value (\w+-\w+)', query)
        if match:
            product_code = match.group(1)
            for item in CleanedDataSet:
                if item["question"].endswith(product_code):
                    print('item["answer"]',item["answer"])
                    return item["answer"]
            return "Sorry, I couldn't find the total value for the specified product code."
        else:
            return answers[answer_index]

# Load QA pairs and preprocess data only once
# qa_pairs = CleanedDataSet()
# corpus, questions, answers = preprocess_data(qa_pairs)

# TF-IDF vectorization
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(corpus)


# response = get_answer(question, vectorizer, tfidf_matrix, questions, answers)

# def generate_response(prompt):
#     # Load pre-trained model and tokenizer
#     model_name = "gpt2"
#     tokenizer = GPT2Tokenizer.from_pretrained(model_name)
#     model = GPT2LMHeadModel.from_pretrained(model_name)

#     # Add a padding token to the tokenizer
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})

#     # Tokenize the prompt with attention mask
#     prompt = prompt + '''\nGenerate the proper sentence:'''
#     input_ids = tokenizer.encode(prompt, return_tensors="pt", padding=True, truncation=True)

#     # Generate completion
#     output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)

#     # Decode the generated output
#     generated_sentence = tokenizer.decode(output[0], skip_special_tokens=True)

#     # Extracting only the relevant sentence
#     generated_sentence = generated_sentence.split('\n\n')[3]

#     return generated_sentence

# # Example usage
# q = 'What is the product code for Panel (Meter) Cover Plastic?\n'
# a =  '\nAnswer: 2318A-RESW'
# prompt = q + a
# response = generate_response(prompt)
# print(response)
