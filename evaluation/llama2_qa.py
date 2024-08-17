from __future__ import absolute_import, division, print_function
import time
import os
import sys
import tqdm
import json
import argparse
import logging
import requests

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S %Z')
logger = logging.getLogger(__name__)

def rmANS(response):
    if "[ANS]" in response:
        context = response.split("[ANS] ")[1].split(" [/ANS]")[0]
        if "[ANS]" in context:
            context = response.split(" [ANS]")[0]
            return context
        else:
            return context
    else:
        return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LLaMA2 QA')
    parser.add_argument(
        "--url",
        default="http://localhost:8888/v1/chat/completions",
        help="url for calling localhost api",
    )
    parser.add_argument(
        "--content_type",
        default="application/json",
        help="content type for api headers",
    )
    parser.add_argument(
        "--input_path",
        default="./model/out/QG-DPO-dynamic-ACE2005-Llama-2-7b",
        help="path to input file for qg",
    )
    parser.add_argument(
        "--output_path",
        default="./evaluation/out",
        help="path to output",
    )
    parser.add_argument(
        "--num_shots",
        default="5",
        help="number of shots",
    )
    parser.add_argument(
        "--model_name",
        default="Llama2-13b-Chat",
        help="model name",
    )

    args = parser.parse_args()

    input_file = args.input_path + "/generated_predictions.json"
    input_name = args.input_path.split("QG-")[1]
    aim_path = args.output_path + "/QA-" + args.model_name + "-" + input_name + '.json'

    with open(args.input_dir) as file:
        data = json.load(file)
        file.close()

    shots = [
        [
            {"role": "user", "content": "question: Who made the battle in Baghdad? "
                                        "context: US Secretary of Defense Donald Rumsfeld dismissed worries that there were insufficient forces in the Gulf region if the battle for Baghdad goes wrong."},
            {"role": "assistant", "content": "[ANS] US [/ANS]"}
        ],
        [
            {"role": "user", "content": "question: Who was nominated? "
                                        "context: Senator Christopher Dodd of Connecticut made the announcement today that he would not be the 10th candidate for the nomination."},
            {"role": "assistant", "content": "[ANS] candidate [/ANS]"}
        ],
        [
            {"role": "user", "content": "question: Who was fired by Maine? "
                                        "context: We're talking about possibilities of full scale war with former Congressman Tom Andrews, Democrat of Maine."},
            {"role": "assistant", "content": "[ANS] Tom Andrews [/ANS]"}
        ],
        [
            {"role": "user", "content": "question: Who died that cause Clinton suffered greatly? "
                                        "context: Clinton suffered greatly over the 19 Rangers that died, 18 on the 3rd of October and Matt Reersen (ph) three days later."},
            {"role": "assistant", "content": "[ANS] Rangers, Matt Reersen [/ANS]"}
        ],
        [
            {"role": "user", "content": "question: Where did the election takes place? "
                                        "context: He lost an election to a dead man."},
            {"role": "assistant", "content": "[ANS] None [/ANS]"}
        ]
    ]

    headers = {"Content-Type": args.content_type}
    output = []
    for element in tqdm.tqdm(data):
        context = element['context']
        question = element['question']
        if int(args.num_shots) == 0:
            messages = {
                "model": "llama2-13b-chat",
                "messages": [
                    {"role": "system", "content": "You are a precise and concise assistant. "
                                                  "Your task is to extract some words base directly on the "
                                                  "provided context to answer the given questions. "
                                                  "Please wrap your answer with the following tags [ANS] [/ANS]. "
                                                  "If a question has multiple correct answers within the context, "
                                                  "list them all, separated by commas. "
                                                  "If there is no answer in the context, just reply [ANS] None [/ANS]. "
                                                  "Do NOT add any introductory phrases, "
                                                  "explanations, or additional information "
                                                  "outside of the given context."},
                    {"role": "user", "content": "question: " + question + " "
                                                "context: " + context}
                ],
                "repetition_penalty": 1.0,
            }
        else:
            messages = {
                "model": "llama2-13b-chat",
                "messages": [
                    {"role": "system", "content": "You are a precise and concise assistant. "
                                                  "Your task is to extract some words base directly on the "
                                                  "provided context to answer the given questions. "
                                                  "Please wrap your answer with the following tags [ANS] [/ANS]. "
                                                  "If a question has multiple correct answers within the context, "
                                                  "list them all, separated by commas. "
                                                  "If there is no answer in the context, just reply [ANS] None [/ANS]. "
                                                  "Do NOT add any introductory phrases, "
                                                  "explanations, or additional information "
                                                  "outside of the given context."}
                ],
                "repetition_penalty": 1.0
            }
            for i in range(int(args.num_shots)):
                messages["messages"].append(shots[i][0])
                messages["messages"].append(shots[i][1])
            messages["messages"].append({"role": "user", "content": "question: " + question + " "
                                                                    "context: " + context})

        response = requests.post(args.url, json=messages, headers=headers)
        response = json.loads(response.text)['choices'][-1]['message']['content']
        try:
            element['response'] = rmANS(response)
        except:
            print(response)
            element['response'] = response
        output.append(element)

    with open(aim_path, 'w', encoding='utf-8') as file:
        json.dump(output, file, indent=4)
        file.close()
    print("llama2 qa finished")