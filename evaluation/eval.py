from __future__ import absolute_import, division, print_function
import time
import os
import sys
import json
import math
import argparse
import logging

from utils import exact_match_accuracy
from utils import content_overlap_ratio
from utils import semantic_similarity
from utils import print_data

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S %Z')
logger = logging.getLogger(__name__)

def get_wa_data(data):
    wa_data = []
    for item in data:
        if item["answer"] == "None":
            continue
        else:
            wa_data.append(item)
    return wa_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='eval-qa')
    parser.add_argument(
        "--input_dir",
        default=".evaluation/out/QG-DPO-dynamic-ACE2005-Llama-2-7b",
        help="path to qa input file",
    )
    args = parser.parse_args()

    with open(args.input_dir) as file:
        data = json.load(file)
        file.close()

    wa_data = get_wa_data(data)

    score_lists = [exact_match_accuracy(wa_data), content_overlap_ratio(wa_data), semantic_similarity(wa_data),
        exact_match_accuracy(data), content_overlap_ratio(data), semantic_similarity(data)]

    print_data(score_lists)