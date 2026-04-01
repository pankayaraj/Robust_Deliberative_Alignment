from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
import argparse
from tqdm import tqdm
import torch
import gc
import numpy as np
from trl import SFTConfig, SFTTrainer
import wandb

from peft import LoraConfig,PeftConfig, PeftModel,inject_adapter_in_model, get_peft_model
import pandas as pd

parser = argparse.ArgumentParser(description='Generation Arguments')
parser.add_argument("--dir", type=str, default="results/MODEL_Qwen2.5-1.5B-Instruct_DATASET_STAR-41K-DA-Filtered-DeepSeek-R1-Distill-Qwen-32B/3")
parser.add_argument("--hf_token", type=str, default="")
parser.add_argument("--batch_size", type=int, default=16)

parser.add_argument("--max_new_tokens", type=int, default=2048)
parser.add_argument("--max_length", type=int, default=8192)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=1.0)

args = parser.parse_args()


def get_chat_template(model_name):
    if "qwen" in model_name.lower():
        PROMPT_TEMPLATE = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nQ: {prompt}<|im_end|>\n<|im_start|>assistant\nAnswer:."
        COMPLETION_TEMPLATE = "<think>{response}<|im_end|>\n"
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant"
    elif "llama" in model_name.lower():
        PROMPT_TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\nQ: {prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAnswer:"
        COMPLETION_TEMPLATE = "<think>{response}<|eot_id|>"
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>"
    elif "gemma" in model_name.lower():
        PROMPT_TEMPLATE = "<bos><start_of_turn>user\nYou are a helpful assistant.\n\nQ: {prompt}<end_of_turn><start_of_turn>model\nAnswer:"
        COMPLETION_TEMPLATE = "<think>{response}<end_of_turn>\n"
        instruction_template = "<start_of_turn>user"
        response_template = "<start_of_turn>model"
    
    return PROMPT_TEMPLATE, COMPLETION_TEMPLATE



import csv
def save_prompt_response_csv(list_a, list_b, list_c, list_d, filename, col1="prompt", col2="golden_answer", col3="response", col4="category"):
    seperator = "__SEPERATOR__"
    list_d = [seperator.join(l_d) for l_d in list_d]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([col1, col2, col3, col4])   # header
        for a, b, c, d in zip(list_a, list_b, list_c, list_d):
            writer.writerow([a, b, c, d])
            

model_dir = args.dir.replace("/results/", "/output/")

tokenizer = AutoTokenizer.from_pretrained(model_dir, token=args.hf_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = PeftConfig.from_pretrained(model_dir)

model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto", dtype=torch.bfloat16, token=args.hf_token)
model.config.use_cache = False
if "DeepSeek" in model_dir and "DeepSeek" not in config.base_model_name_or_path:
    model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, model_dir)
model = model.merge_and_unload()


generation_kwargs = {
                    "temperature":args.temperature,
                    "top_p":args.top_p,
                    "do_sample": True,
                    "max_new_tokens":args.max_new_tokens,
                    "max_length": args.max_length,
                    # "stop_strings":["/n/n"],
                    # "tokenizer":tokenizer
                }


MMLU_1 = load_dataset("cais/mmlu", "abstract_algebra")["test"]
MMLU_2 = load_dataset("cais/mmlu", "computer_security")["test"]
MMLU_3 = load_dataset("cais/mmlu", "business_ethics")["test"]
MMLU_4 = load_dataset("cais/mmlu", "high_school_chemistry")["test"]
MMLU_5 = load_dataset("cais/mmlu", "college_physics")["test"]
MMLU_6 = load_dataset("cais/mmlu", "anatomy")["test"]
MMLU_7 = load_dataset("cais/mmlu", "machine_learning")["test"]
MMLU_8 = load_dataset("cais/mmlu", "econometrics")["test"]

MMLU_LIST  = [MMLU_1, MMLU_2, MMLU_3, MMLU_4, MMLU_5, MMLU_6, MMLU_7, MMLU_8]
MMLU_SPLITS = ["abstract_algebra", "computer_security", "business_ethics", "high_school_chemistry", "college_physics", "anatomy", "machine_learning", "econometrics"]

MMLU = {
    "prompt":[],
    "answer":[],
    "category":[],
}
template = """Answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D, E]
Give me the answer only within \\boxed[]. For instance of your answer is D the string you should give is \\boxed[ D ]. If your answer is C the string you should give is \\boxed[ C ].
Question: {prompt}\n
A) {a}\n
B) {b}\n
C) {c}\n
D) {d}\n
"""
for name, mmlu in zip(MMLU_SPLITS, MMLU_LIST):
    for data in mmlu:
        MMLU["prompt"].append(
            template.format(
                prompt=data["question"],
                a=data["choices"][0],
                b=data["choices"][1],
                c=data["choices"][2],
                d=data["choices"][3],
                )
        )

        MMLU["answer"].append(data["answer"])
        MMLU["category"].append(data["subject"])
MMLU = Dataset.from_dict(MMLU)
        

batch_size = args.batch_size
num_generations = 8

PROMPT_TEMPLATE, COMPLETION_TEMPLATE = get_chat_template(config.base_model_name_or_path)

RESPONSES = []


PROMPTS = MMLU["prompt"]
GOLDEN_ANSWERS = [str(d) for d in MMLU["answer"]]
CATEGORY =  [str(c) for c in MMLU["category"]]

for idx in tqdm(range(0, len(PROMPTS), batch_size)):    
    
    start = idx
    end = min(len(PROMPTS), idx + batch_size)

    questions = PROMPTS[start:end]
    questions = [PROMPT_TEMPLATE.format(prompt=q) for q in questions]
    RES = [[] for _ in range(len(questions))]
    questions_multiples = questions*num_generations

    inputs = tokenizer(questions_multiples, return_tensors="pt", max_length=args.max_length, padding=True, truncation=True).to(model.device)
    outputs = model.generate(**inputs, **generation_kwargs)

    max_len = inputs["input_ids"].shape[1]
    responses = tokenizer.batch_decode(outputs[:,max_len:], skip_special_tokens=True)

    for i, r in enumerate(responses):
        RES[i%(end-start)].append(str(r))
        
    RESPONSES += RES

    del inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()


save_prompt_response_csv(
        PROMPTS,
        GOLDEN_ANSWERS,
        CATEGORY,
        RESPONSES,
        args.dir + "/mulit_generation_MMLU.csv"
)



    




