from sklearn.metrics.pairwise import cosine_similarity
from qapairs import CleanedDataSet

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
        return answers[question_index]
    else:
        # Check if the query contains 'total value' and a product code
        match = re.search(r'total value (\w+-\w+)', query)
        if match:
            product_code = match.group(1)
            for item in CleanedDataSet:
                if item["question"].endswith(product_code):
                    return item["answer"]
            return "Sorry, I couldn't find the total value for the specified product code."
        else:
            return answers[answer_index]
