from __future__ import absolute_import, division, print_function
import json
import argparse
import logging
import hashlib

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S %Z')
logger = logging.getLogger(__name__)

def generate_sha1(path):
    sha1 = hashlib.sha1()
    with open(path, 'rb') as file:
        while True:
            data = file.read(8192)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()

def merge_data(path, type):
    outputs = []
    for name in ['train', 'dev']:
        file_path = path + "/" + name + f"_sft-{type}.json"
        with open(file_path) as file:
            output = json.load(file)
        outputs += output

    return outputs

def generate_description(path, type):
    sha1 = generate_sha1(path)
    description = {
        f"{type}_qg":
            {
                "file_name": f"{type}_qg.json",
                "file_sha1": sha1
            }
        }

    return description

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='merge train and dev')
    parser.add_argument(
        "--sft_path",
        default="../data",
        help="path to training file",
    )
    parser.add_argument(
        "--constructed_path",
        default="../data",
        help="path to output merge data",
    )
    parser.add_argument(
        "--type",
        default="template",
        help="",
    )

    args = parser.parse_args()

    merge_data = merge_data(args.sft_path, args.type)

    with open(args.constructed_path + f"/{args.type}_qg.json", 'w', encoding='utf-8') as data_file:
        json.dump(merge_data, data_file, indent=4)
        data_file.close()

    if args.type == 'template':
        data_info = generate_description(args.constructed_path + f"/{args.type}_qg.json", args.type)

        with open(args.constructed_path + f"/dataset_info_{args.type}_qg.json", 'w', encoding='utf-8') as info_file:
            json.dump(data_info, info_file, indent=4)
            info_file.close()
