import json
import os
import sys
import logging
import argparse
import spacy
from tqdm import tqdm
from multiprocessing import Pool

def calculate_similarity(text1, text2, nlp):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

def process_item(item):
    nlp = spacy.load('en_core_web_lg')
    context = item["input"].split("context: ")[1].replace(" * ", " ").replace(" *", "").replace("* ", "").replace(" ", "")
    answer = item["answer"]
    reclist = []
    qalist = []
    for rec in item["recover"]:
        similarity_score = calculate_similarity(rec, context, nlp)
        reclist.append(similarity_score)
    for resp in item["response"]:
        similarity_score = calculate_similarity(resp, answer, nlp)
        qalist.append(similarity_score)
    item["rec-score"] = reclist
    item["qa-score"] = qalist
    return item

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Selecting Questions')
    parser.add_argument('--test', action='store_true')
    parser.add_argument(
        "--input_dir",
        default="",
    )
    parser.add_argument(
        "--output_dir",
        default="",
    )
    parser.add_argument(
        "--alpha",
        default=0.3,
    )
    parser.add_argument(
        "--beta",
        default=0.7,
    )
    args = parser.parse_args()

    with open(args.input_dir) as file:
        data = json.load(file)
        file.close()

    with Pool(processes=16) as pool:
        results = []
        for result in tqdm(pool.imap_unordered(process_item, data), total=len(data)):
            results.append(result)

    out = []
    aim_path = args.output_dir + "/rewardRAMS-question-a" + str(args.alpha) + "-b" + str(args.beta) + ".json"
    for item in results:
        recover_score = item["rec-score"]
        qa_score = item["qa-score"]

        score = []
        for rec, qa in zip(recover_score, qa_score):
            final_score = float(args.alpha) * float(rec) + float(args.beta) * float(qa)
            score.append(final_score)

        max_score_index = score.index(max(score))
        min_score_index = score.index(min(score))

        if max(score) < 0.6 or max(score) - min(score) < 0.5:
            continue

        chosen_question = item["predict"][max_score_index]
        reject_question = item["predict"][min_score_index]
        ele = {"instruction": item["instruction"], "input": item["input"],
               "output": [chosen_question, reject_question]}
        out.append(ele)

    with open(aim_path, 'w', encoding='utf-8') as file:
        json.dump(out, file, indent=4)
        file.close()