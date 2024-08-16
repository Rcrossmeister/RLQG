import json
import os
import argparse

def read_jsonl(file_path):
    with open(file_path) as file:
        return [json.loads(line.strip()) for line in file]

def read_json(file_path):
    with open(file_path) as file:
        return json.load(file)

def write_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)

def QG2IPMnQA(seq, instruction, bs_path, ref_path, output_paths):
    data = read_jsonl(bs_path)
    aimset = read_json(ref_path)
    
    grouped_data = [ [d['predict'] for d in data[i:i+seq]] for i in range(0, len(data), seq) ]
    
    ip_input_out = []
    qa_input_out = []
    
    for item, element in zip(grouped_data, aimset):
        element['predict'] = item
        
        for i in range(seq):
            ip_input_ele= {
                "instruction": instruction,
                "input": f"trigger: {element['input'].split('trigger:')[1].split('context:')[0].replace(' ', '')} question: {element['predict'][i]}",
                "output": ""
            }
            ip_input_out.append(ip_input_ele)
            
            qa_input_ele = {
                "context": element['input'].split("context: ")[1].replace(" </s>", "").replace("* ", "").replace(" *", ""),
                "answer": element['answer'],
                "question": element['predict'][i]
            }
            qa_input_out.append(qa_input_ele)
    
    write_json(ip_input_out, output_paths["ip_input"])
    write_json(qa_input_out, output_paths["qa_input"])

def IPMnQA2RW(seq, beams_path, qa_path, output_paths):
    beams_data = read_jsonl(beams_path)
    qa_data = read_json(qa_path)
    aimset = read_json(aimset_path)
    
    grouped_beams_data = [ [d['predict'] for d in beams_data[i:i+seq]] for i in range(0, len(beams_data), seq) ]
    grouped_qa_data = [ [d['response'] for d in qa_data[i:i+seq]] for i in range(0, len(qa_data), seq) ]
    
    collect_input_out = []
    for beams, qa, item in zip(grouped_beams_data, grouped_qa_data, aimset):
        item['recover'] = beams
        item['response'] = qa
        collect_input_out.append(item)
    
    write_json(collect_input_out, output_path["collect_input"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert File")
    parser.add_argument("stage", choices=["QG2IPMnQA", "IPMnQA2RW"], help="Select the stage.")
    parser.add_argument("--bs", default="./out/BS-SFT-dynamic-ACE2005-Llama-2-7b")
    parser.add_argument("--ref", default="./data/REF-dynamic-ACE2005.json")
    parser.add_argument("--ip", default="./out/IP-BS-SFT-dynamic-ACE2005-Llama-2-7b")
    parser.add_argument("--qa", default="./out/QA-BS-SFT-dynamic-ACE2005-Llama-2-7b")
    args = parser.parse_args()

    seq = 5
    instruction = ("You are a creative assistant. Based on the provided question and event trigger by user, "
                   "reconstruct the context of the event with as much detail and clarity as possible. Ensure your response "
                   "captures the essence of the situation, including relevant key elements and implications related to the trigger "
                   "and question. Your response should always include the given original trigger. Use 'somewhere' for 'Where' questions, "
                   "'someone' for 'Who' questions, and 'Something' for 'What' questions.")

    bs_path = args.bs + "/generated_predictions.jsonl"
    ref_path = args.ref
    ip_path = args.ip + "/generated_predictions.jsonl"
    qa_path = args.qa + ""

    output_paths = {
        "ip_input": "",
        "qa_input": "",
        "collect_input": "",
    }

    # Run the selected stage
    if args.stage == "QG2IPMnQA":
        QG2IPMnQA(args.seq, instruction, bs_path, ref_path , output_paths)
    elif args.stage == "IPMnQA2RW":
        IPMnQA2RW(args.seq, ip_path, qa_path, output_paths)