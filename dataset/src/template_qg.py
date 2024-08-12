from __future__ import absolute_import, division, print_function
import time
import os
import sys
import math
import argparse
import logging

current_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(root_path)

from utils import loader
from utils.processors import dataset_processor

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S %Z')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Template Question Generation')
    parser.add_argument('--lower_case', action='store_true', help='lower case')
    parser.add_argument(
        "--file_path",
        default="./ACE2005/data/converted-data",
        help="path to after processed ace file"
    )
    parser.add_argument(
        "--event_template_doc",
        default="./ACE2005/ace_templates",
        help="path to ACE event template file"
    )
    parser.add_argument(
        "--template_type",
        default="dynamic",
        help="question template type"
    )
    parser.add_argument(
        "--sft_output_path",
        default="../model/data",
        help="path to output the llm-sft file"
    )

    args = parser.parse_args()
    ace_processor = dataset_processor(args)

    arg_question_dic = loader.load_template(args.event_template_doc + "/" + args.template_type + ".tsv")
    ace_processor.load_data(args.file_path)

    ace_processor.construct_dataset_qga(arg_question_dic, lower_case=args.lower_case, template_type=args.template_type)

    ace_processor.convert_sft_format(args.sft_output_path, template_type=args.template_type)

    print(f"{args.template_type} template question succesfully saved to {args.sft_output_path}.")
