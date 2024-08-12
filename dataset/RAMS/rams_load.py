import json
import re
import argparse

eng_pattern = r'[a-zA-Z]+'

def process_sentence(sentence):
    return_sentence = ''
    for i, token in enumerate(sentence):
        if token in [',', '.', ':', ';', '!', '?', '"'] or (
                i > 0 and sentence[i - 1] in ["'", "n’t", "’s", "’re", "’ve", "’m", "’ll", "’d"]):
            return_sentence += token
        else:
            return_sentence += ' ' + token
    return return_sentence.strip().replace("“", " “").replace("”", "” ").replace(" ’", "’")

parser = argparse.ArgumentParser(description='RAMS Pre-processing')
parser.add_argument("input_path", help="Path for input directory.")
args = parser.parse_args()

with open('./rams_templates/standard.json', 'r', encoding='utf-8') as file:
    pronoun = json.load(file)

outputs = []

stage = ["train", "dev"]

for s in stage:
    data = []
    with open(args.input_path + f"/data/{s}.jsonlines", 'r', encoding='utf-8') as file:
        for line in file:
            data.append(line.strip())

    for element in data:
        item = json.loads(element)

        if not item["ent_spans"]:
            continue

        sentences_list = item["sentences"]
        sentence_continuous = []
        context_token = []
        for sentence_split in sentences_list:
            complete_sentence = process_sentence(sentence_split)
            sentence_continuous.append(complete_sentence)
            for token in sentence_split:
                context_token.append(token)

        context = re.sub(r'[^\x00-\x7F]+', '', (" ".join(sentence_continuous)))
        context = context.replace("  ", " ")

        event_idx = item["gold_evt_links"][0][0]
        trigger = context_token[event_idx[0]:event_idx[1]+1][0]

        arg_role_dict = {}
        for event_item in item["gold_evt_links"]:
            arg_idx = event_item[1]
            arg = " ".join(context_token[arg_idx[0]:arg_idx[1]+1])
            link_str = event_item[-1]
            role = re.findall(eng_pattern, link_str)[-1]

            arg_role_dict.setdefault(role, []).append(arg)

        for role in arg_role_dict.keys():

            role_processed, interrogative_pronoun = pronoun[role][0], pronoun[role][1]

            if interrogative_pronoun == "Where":
                question = interrogative_pronoun + " the " + trigger + " takes place?"
            else:
                question = interrogative_pronoun + " is the " + role_processed + " in " + trigger + "?"

            output = {
                "instruction": "",
                "input": "role: " + role_processed + " trigger: " + trigger + " context: " + context.replace(trigger, "* " + trigger + " *"),
                "output": question
            }
            outputs.append(output)

with open("../../model/data/SFT-standard-RAMS.json", 'w', encoding='utf-8') as file:
    json.dump(outputs, file, indent=4)