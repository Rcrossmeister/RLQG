#!/bin/bash

# Locally deploy an off-the-shelf LLMs
python src/api/openai_api_server.py \
    --base_model ../backbone_model/Llama-2-13b-chat-hf \
    --gpus 1,2 \

mkdir out

CUDA_VISIBLE_DEVICES=0

# Supervised fine-tune the backbone model over template questions
python bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path ../backbone_model/Llama-2-7b-hf \
    --dataset SFT-dynamic-ACE2005 \
    --template qg \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ./out/SFT-dynamic-ACE2005-Llama-2-7b \
    --overwrite_cache \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --plot_loss \
    --bf16

# Generate question candidates via beam search augmentation
python bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path ../backbone_model/Llama-2-7b-hf \
    --adapter_name_or_path ./out/SFT-dynamic-ACE2005-Llama-2-7b \
    --dataset SFT-dynamic-ACE2005 \
    --template qg \
    --finetuning_type lora \
    --output_dir ./out/BS-SFT-dynamic-ACE2005-Llama-2-7b \
    --per_device_eval_batch_size 1 \
    --max_samples 20000 \
    --predict_with_generate \
    --bf16 \
    --do_sample False \
    --num_return_sequences 5 \
    --num_beams 10

python converter.py \
    --stage QG2IPMnQA \
    --bs ./out/BS-SFT-dynamic-ACE2005-Llama-2-7 \
    --ref ./data/REF-dynamic-ACE2005.json

# The inverse prompting process
python bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path ../backbone_model/Llama-2-13b-hf \
    --adapter_name_or_path ../backbone_model/IPM-Llama-2-13b \
    --dataset IP-dynamic-ACE2005 \
    --template default \
    --finetuning_type lora \
    --output_dir ./out/IP-BS-SFT-dynamic-ACE2005-Llama-2-7b \
    --per_device_eval_batch_size 1 \
    --max_samples 20000 \
    --predict_with_generate \
    --bf16

# The question answering process
python ../evaluation/llama2_qa.py \
    --input_dir ../evaluation/questions/BS-SFT-dynamic-ACE2005-Llama-2-7b.json \
    --output_dir ./out/QA-BS-SFT-dynamic-ACE2005-Llama-2-7b \
    --num_shots 5 \
    --url http://localhost:19777/v1/chat/completions

python converter.py \
    --stage IPMnQA2RW \
    --ip ./out/IP-BS-SFT-dynamic-ACE2005-Llama-2-7b \
    --qa ./out/QA-BS-SFT-dynamic-ACE2005-Llama-2-7b

python collector.py \
    --input_path ./out/RW-dynamic-ACE2005-Llama-2-7b.json \
    --output_path ./data/PPO-dynamic-ACE2005.json

# DPO training with the QA and IPM rewards
python bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path ../backbone_model/Llama-2-7b-hf \
    --adapter_name_or_path ./out/SFT-dynamic-ACE2005-Llama-2-7b \
    --create_new_adapter \
    --dataset PPO-dynamic-ACE2005 \
    --template qg \
    --finetuning_type lora \
    --lora_target q_proj,v_proj \
    --output_dir ./out/DPO-dynamic-ACE2005-Llama-2-7b \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 1e-5 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --overwrite_output_dir \
    --num_beams 10 \
    --bf16

# Get the generated questions by DPO refined model
python bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path ../backbone_model/Llama-2-7b-hf \
    --adapter_name_or_path ./out/SFT-dynamic-ACE2005-Llama-2-7b,./out/DPO-dynamic-ACE2005-Llama-2-7b \
    --dataset QG-dynamic-ACE2005 \
    --template qg \
    --finetuning_type lora \
    --output_dir ./out/QG-DPO-dynamic-ACE2005-Llama-2-7b \
    --per_device_eval_batch_size 1 \
    --max_samples 10000 \
    --predict_with_generate \
    --bf16