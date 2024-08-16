from __future__ import absolute_import, division, print_function
from collections import Counter
import json
import logging
import os
import itertools
from .loader import (
    convert_examples_to_features_qga,
    load_sentences
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S %Z",
)

logger = logging.getLogger(__name__)

class dataset_processor:
    def __init__(self, args):
        self.args = args
        self.train_sentences = []
        self.dev_sentences = []
        self.test_sentences = []
        self.train_annotations = []
        self.dev_annotations = []
        self.test_annotations = []
        self.train_map_lists = []
        self.dev_map_lists = []
        self.test_map_lists = []

    def load_data(self, file_path):
        train_path = file_path + "/train_convert.json"
        dev_path = file_path + "/dev_convert.json"
        test_path = file_path + "/test_convert.json"
        self.train_samples = load_sentences(train_path)
        self.dev_samples = load_sentences(dev_path)
        self.test_samples = load_sentences(test_path)

    def generate_vocab(self, files_list):
        self.category_to_index = dict()
        self.index_to_category = dict()
        self.counter_event = Counter()

        self.category_to_index["O"] = 0
        self.index_to_category[0] = "O"
        for file in files_list:
            with open(file) as f:
                for line in f:
                    example = json.loads(line)
                    labels = example["trigger_label"]
                    for label in labels:
                        if label == "O":
                            continue
                        event_type = label
                        self.counter_event[event_type] += 1
                        if event_type not in self.category_to_index:
                            index = len(self.category_to_index)
                            self.category_to_index[event_type] = index
                            self.index_to_category[index] = event_type

    def construct_dataset_qga(
        self,
        arg_question_dic,
        template_type,
        lower_case=True
    ):
        self.train_dataset = convert_examples_to_features_qga(
            examples=self.train_samples,
            query_templates=arg_question_dic,
            lower_case=lower_case,
            template_type=template_type
        )

        self.dev_dataset = convert_examples_to_features_qga(
            examples=self.dev_samples,
            query_templates=arg_question_dic,
            lower_case=lower_case,
            template_type=template_type
        )

        self.test_dataset = convert_examples_to_features_qga(
            examples=self.test_samples,
            query_templates=arg_question_dic,
            lower_case=lower_case,
            template_type=template_type,
        )

        return (
            self.train_dataset,
            self.dev_dataset,
            self.test_dataset,
        )

    def convert_sft_format(
            self, 
            output_path,
            template_type
    ):
        os.makedirs(output_path, exist_ok=True)
        combined_train_dev_output = []
        ref_output = []
        for dataset, name in [(self.train_dataset, "train"), (self.dev_dataset, "dev"), (self.test_dataset, "test")]:
            if name != "test":
                for input_text, question_text, _ in dataset:
                    template_output = {
                        "instruction": "",
                        "input": input_text.replace(" </s>", ""),
                        "output": question_text.replace(" </s>", "")
                    }
                    ref_ele = {
                        "instruction": "",
                        "input": input_text.replace(" </s>", ""),
                        "output": question_text.replace(" </s>", ""),
                        "answer": (", ".join(_)) if _ else "None"
                    }
                    combined_train_dev_output.append(template_output)
                    ref_output.append(ref_ele)
            else:
                output_list = []
                aim_path = output_path + "/QG-" + template_type + "-ACE2005.json"
                for input_text, question_text, _ in dataset:
                    output = {
                        "instruction": "",
                        "input": input_text.replace(" </s>", ""),
                        "output": question_text.replace(" </s>", "")
                    }
                    output_list.append(output)
                with open(aim_path, 'w', encoding='utf-8') as file:
                    json.dump(output_list, file, indent=4)
                    file.close()
        
        combined_ref_path = output_path + "/REF-" + template_type + "-ACE2005.json"
        combined_aim_path = output_path + "/SFT-" + template_type + "-ACE2005.json"
        with open(combined_aim_path, 'w', encoding='utf-8') as file:
            json.dump(combined_train_dev_output, file, indent=4)
            file.close()
        with open(combined_ref_path, 'w', encoding='utf-8') as file:
            json.dump(ref_output, file, indent=4)
            file.close()