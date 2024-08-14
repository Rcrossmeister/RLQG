import spacy
import json
import nltk
from nltk.tokenize import word_tokenize

def semantic_similarity(data):
    nlp = spacy.load("en_core_web_lg")
    total_similarity = 0
    total_count = len(data)

    for element in data:
        answer_doc = nlp(element["answer"])
        response_doc = nlp(element["response"])
        if answer_doc.vector_norm == 0 or response_doc.vector_norm == 0:
            if element["answer"] == element["response"]:
                similarity = 1
            elif element["answer"] in element["response"]:
                similarity = len(element["answer"]) / len(element["response"])
            else:
                similarity = 0
        else:
            similarity = answer_doc.similarity(response_doc)
        total_similarity += similarity

    average_similarity = total_similarity / total_count if total_count > 0 else 0
    return average_similarity * 100

def content_overlap_ratio(data):
    total_ratio = 0
    total_count = len(data)

    for element in data:
        response = set(word_tokenize(element["response"].lower()))
        answer = set(word_tokenize(element["answer"].lower()))

        intersection = answer.intersection(response)
        ratio = len(intersection) / max(len(answer), len(response))
        total_ratio += ratio

    average_ratio = total_ratio / total_count if total_count > 0 else 0
    return average_ratio * 100

def exact_match_accuracy(data):
    correct_count = 0
    total_count = len(data)

    for element in data:
        response = element["response"].lower()
        answer = element["answer"].lower()
        if response == answer:
            correct_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0
    return accuracy * 100

def print_data(score_lists):
    metrics = ['EM', 'COR', 'SemSim']
    
    print('============================ Practical Eval ============================')
    print("{:<10} {:^20} {:^20} {:^20}".format("Metric", *metrics))
    print('========================================================================')
    print("{:<10} {:^20.2f} {:^20.2f} {:^20.2f}".format("Value(%)", *score_lists[:3]))
    
    print()
    
    print('============================== Full Eval ===============================')
    print("{:<10} {:^20} {:^20} {:^20}".format("Metric", *metrics))
    print('========================================================================')
    print("{:<10} {:^20.2f} {:^20.2f} {:^20.2f}".format("Value(%)", *score_lists[3:]))