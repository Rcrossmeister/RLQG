from __future__ import absolute_import, division, print_function
import time
import os
import sys
import json
import math
import argparse
import logging
import openai
import tqdm

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
    parser = argparse.ArgumentParser(description='API QG')
    parser.add_argument(
        "--api_key",
        default="sk-sOBGld2dsK5J0QhRFf9a82777b96407bBb9b31F154037424",
        help="key for calling api",
    )
    parser.add_argument(
        "--api_base",
        default="https://api.openai.com/v1",
        help="base for calling api",
    )
    parser.add_argument(
        "--api_model",
        default="gpt-4-1106-preview",
        help="model for calling api",
    )
    parser.add_argument(
        "--input_dir",
        default="./model/out/QG-DPO-dynamic-ACE2005-Llama-2-7b",
        help="path to qa input file",
    )
    parser.add_argument(
        "--output_dir",
        default="./evaluation/outs",
        help="path to output",
    )
    parser.add_argument(
        "--num_shots",
        default="5",
        help="number of shots",
    )

    args = parser.parse_args()

    input_file = args.input_dir + "/generated_predictions.json"
    
    

    openai.api_key = args.api_key
    openai.api_base = args.api_base

    input_file = args.input_path + "/generated_predictions.json"
    input_name = args.input_path.split("QG-")[1]
    aim_path = args.output_path + "/QA-" + args.api_model + "-" + input_name + '.json'

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
            {"role": "user", "content": "question: Who is responsible for the transport event? "
                                        "context: Even as the secretary of homeland security was putting his people on high alert last month, a 30-foot Cuban patrol boat with four heavily armed men landed on American shores, utterly undetected by the Coast Guard Secretary Ridge now leads."},
            {"role": "assistant", "content": "[ANS] None [/ANS]"}
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
            {"role": "user", "content": "question: Where did the election takes place? "
                                        "context: He lost an election to a dead man."},
            {"role": "assistant", "content": "[ANS] None [/ANS]"}
        ],
        [
            {"role": "user", "content": "question: Who died that cause Clinton suffered greatly? "
                                        "context: Clinton suffered greatly over the 19 Rangers that died, 18 on the 3rd of October and Matt Reersen (ph) three days later."},
            {"role": "assistant", "content": "[ANS] Rangers, Matt Reersen [/ANS]"}
        ],
        [
            {"role": "user", "content": "question: Where was a pregnant woman and the 13-year-old child killed? "
                                        "context: Eight people, including a pregnant woman and a 13-year-old child were killed in Monday's Gaza raid, provoking US-led international calls for Israeli restraint."},
            {"role": "assistant", "content": "[ANS] Gaza [/ANS]"}
        ]
    ]

    output = []
    for element in tqdm.tqdm(data):
        context = element["context"]
        question = element["question"]

        if int(args.num_shots) == 0:
            messages = [
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
            ]
        else:
            messages = [
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
            ]
            for i in range(int(args.num_shots)):
                messages.append(shots[i][0])
                messages.append(shots[i][1])
            messages.append({"role": "user", "content": "question: " + question + " context: " + context})
        completion = openai.ChatCompletion.create(
            model=args.api_model,
            messages=messages,
        )
        response = completion.choices[0].message.content
        element['response'] = rmANS(response)
        output.append(element)

    with open(aim_path, 'w', encoding='utf-8') as file:
        json.dump(output, file, indent=4)
        file.close()
    print("openai qa finished")