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
parser.add_argument("--batch_size", type=int, default=8)

parser.add_argument("--max_new_tokens", type=int, default=2048)
parser.add_argument("--max_length", type=int, default=8192)
parser.add_argument("--temperature", type=float, default=0.7)
parser.add_argument("--top_p", type=float, default=1.0)

args = parser.parse_args()


def get_chat_template(model_name):
    if "qwen" in model_name.lower():
        PROMPT_TEMPLATE = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        COMPLETION_TEMPLATE = "<think>{response}<|im_end|>\n"
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant"
    elif "llama" in model_name.lower():
        PROMPT_TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        COMPLETION_TEMPLATE = "<think>{response}<|eot_id|>"
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>"
    elif "gemma" in model_name.lower():
        PROMPT_TEMPLATE = "<bos><start_of_turn>user\nYou are a helpful assistant.\n\n{prompt}<end_of_turn><start_of_turn>model\n"
        COMPLETION_TEMPLATE = "<think>{response}<end_of_turn>\n"
        instruction_template = "<start_of_turn>user"
        response_template = "<start_of_turn>model"
    
    return PROMPT_TEMPLATE, COMPLETION_TEMPLATE



import csv
def save_prompt_response_csv(list_a, list_b, list_c, filename, col1="prompt", col2="response", col3="label"):
    seperator = "__SEPERATOR__"
    list_b = [seperator.join(l_b) for l_b in list_b]
    list_c = [seperator.join(l_c) for l_c in list_c]


    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([col1, col2, col3])   # header
        for a, b, c in zip(list_a, list_b, list_c):
            writer.writerow([a, b, c])
            
# save_dir = ""

# from pathlib import Path
# Path(save_dir).mkdir(parents=True, exist_ok=True)

model_dir = args.dir.replace("/results/", "/output/")
from pathlib import Path
Path(model_dir).mkdir(parents=True, exist_ok=True)

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
                }


DAN = pd.read_csv("files/DAN.csv")
Strong_Reject = pd.read_csv("files/strongreject.csv")
Wildjailbreak = pd.read_csv("files/wildjailbreak.csv")


DATASETS = [1,1,1]
batch_size = args.batch_size
num_generations = 8

PROMPT_TEMPLATE, COMPLETION_TEMPLATE = get_chat_template(config.base_model_name_or_path)

WILDJAILBREAK_RESPONSES = []
STRONGREJCT_RESPONSES = []
DAN_RESPONSES = []

if DATASETS[0] == 1:
    PROMPTS = Wildjailbreak["prompt"]

    for idx in tqdm(range(0, len(PROMPTS), batch_size)):    
    
        start = idx
        end = min(len(PROMPTS), idx + batch_size)

        questions = PROMPTS[start:end]
        questions = [PROMPT_TEMPLATE.format(prompt=q) + "<think>" for q in questions]
        RES = [[] for _ in range(len(questions))]
        questions_multiples = questions*num_generations

        inputs = tokenizer(questions_multiples, return_tensors="pt", max_length=args.max_length, padding=True, truncation=True).to(model.device)
        outputs = model.generate(**inputs, **generation_kwargs)

        max_len = inputs["input_ids"].shape[1]
        responses = tokenizer.batch_decode(outputs[:,max_len:], skip_special_tokens=True)

        for i, r in enumerate(responses):
            RES[i%(end-start)].append(str(r))
        
        WILDJAILBREAK_RESPONSES += RES

        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()





if DATASETS[1] == 1:
    PROMPTS = Strong_Reject["prompt"]

    for idx in tqdm(range(0, len(PROMPTS), batch_size)):    
    
        start = idx
        end = min(len(PROMPTS), idx + batch_size)

        questions = PROMPTS[start:end]
        questions = [PROMPT_TEMPLATE.format(prompt=q) + "<think>" for q in questions]
        RES = [[] for _ in range(len(questions))]
        questions_multiples = questions*num_generations

        inputs = tokenizer(questions_multiples, return_tensors="pt", max_length=args.max_length, padding=True, truncation=True).to(model.device)
        outputs = model.generate(**inputs, **generation_kwargs)

        max_len = inputs["input_ids"].shape[1]
        responses = tokenizer.batch_decode(outputs[:,max_len:], skip_special_tokens=True)

        for i, r in enumerate(responses):
            RES[i%(end-start)].append(str(r))
        
        STRONGREJCT_RESPONSES += RES

        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()



if DATASETS[2] == 1:
    PROMPTS = DAN["prompt"]

    for idx in tqdm(range(0, len(PROMPTS), batch_size)):    
    
        start = idx
        end = min(len(PROMPTS), idx + batch_size)

        questions = PROMPTS[start:end]
        questions = [PROMPT_TEMPLATE.format(prompt=q) + "<think>" for q in questions]
        RES = [[] for _ in range(len(questions))]
        questions_multiples = questions*num_generations

        inputs = tokenizer(questions_multiples, return_tensors="pt", max_length=args.max_length, padding=True, truncation=True).to(model.device)
        outputs = model.generate(**inputs, **generation_kwargs)

        max_len = inputs["input_ids"].shape[1]
        responses = tokenizer.batch_decode(outputs[:,max_len:], skip_special_tokens=True)

        for i, r in enumerate(responses):
            RES[i%(end-start)].append(str(r))
        
        DAN_RESPONSES += RES

        del inputs, outputs
        torch.cuda.empty_cache()
        gc.collect()


del model
torch.cuda.empty_cache()
gc.collect()



filter_model_id = "meta-llama/Llama-Guard-3-8B"
filter_tokenizer = AutoTokenizer.from_pretrained(filter_model_id , token=args.hf_token)
if filter_tokenizer.pad_token is None:
    filter_tokenizer.pad_token = filter_tokenizer.eos_token
filter_model = AutoModelForCausalLM.from_pretrained(filter_model_id, torch_dtype=torch.bfloat16, device_map="auto", token=args.hf_token)
 

if DATASETS[0] == 1:
    RESPONSES = WILDJAILBREAK_RESPONSES
    WILDJAILBREAK_SAFETY_LABELS = []

    for idx, responses in tqdm(enumerate(RESPONSES)):
        p = PROMPTS[idx]

        for i, r in enumerate(responses):
            if r == "":
                responses[i] = "       "
        text = [
                filter_tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": str(p)},
                        {"role": "assistant", "content": str(r)},
                    ],
                    tokenize=False
                ) for r in responses
            ]
        model_inputs = filter_tokenizer(responses, padding=True, truncation=True, return_tensors="pt").to(filter_model.device)
        output_ids = filter_model.generate(
                **model_inputs,
                max_new_tokens=100,
        )

        output_ids = output_ids[:, len(model_inputs.input_ids[0]):]
        content = filter_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        safety_labels = ["1" if c.strip().lower().startswith("safe") else "0" for c in content] 
        WILDJAILBREAK_SAFETY_LABELS.append(safety_labels)

        del model_inputs, output_ids
        torch.cuda.empty_cache()
        gc.collect()

    save_prompt_response_csv(
        PROMPTS,
        WILDJAILBREAK_RESPONSES,
        WILDJAILBREAK_SAFETY_LABELS,
        args.dir + "/mulit_generation_wildjailbreak.csv"
    )

if DATASETS[1] == 1:
    RESPONSES = STRONGREJCT_RESPONSES
    STRONGREJCT_SAFETY_LABELS = []

    for idx, responses in tqdm(enumerate(RESPONSES)):
        p = PROMPTS[idx]

        for i, r in enumerate(responses):
            if r == "":
                responses[i] = "       "
        text = [
                filter_tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": str(p)},
                        {"role": "assistant", "content": str(r)},
                    ],
                    tokenize=False
                ) for r in responses
            ]
        model_inputs = filter_tokenizer(responses, padding=True, truncation=True, return_tensors="pt").to(filter_model.device)
        output_ids = filter_model.generate(
                **model_inputs,
                max_new_tokens=100,
        )

        output_ids = output_ids[:, len(model_inputs.input_ids[0]):]
        content = filter_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        safety_labels = ["1" if c.strip().lower().startswith("safe") else "0" for c in content] 

        STRONGREJCT_SAFETY_LABELS.append(safety_labels)

        del model_inputs, output_ids
        torch.cuda.empty_cache()
        gc.collect()
    
    save_prompt_response_csv(
        PROMPTS,
        STRONGREJCT_RESPONSES,
        STRONGREJCT_SAFETY_LABELS,
        args.dir + "/mulit_generation_strongreject.csv"
    )

if DATASETS[2] == 1:
    RESPONSES = DAN_RESPONSES
    DAN_SAFETY_LABELS = []

    for idx, responses in tqdm(enumerate(RESPONSES)):
        p = PROMPTS[idx]

        for i, r in enumerate(responses):
            if r == "":
                responses[i] = "       "
        text = [
                filter_tokenizer.apply_chat_template(
                    [
                        {"role": "user", "content": str(p)},
                        {"role": "assistant", "content": str(r)},
                    ],
                    tokenize=False
                ) for r in responses
            ]
        model_inputs = filter_tokenizer(responses, padding=True, truncation=True, return_tensors="pt").to(filter_model.device)
        output_ids = filter_model.generate(
                **model_inputs,
                max_new_tokens=100,
        )

        output_ids = output_ids[:, len(model_inputs.input_ids[0]):]
        content = filter_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        safety_labels = ["1" if c.strip().lower().startswith("safe") else "0" for c in content] 

        DAN_SAFETY_LABELS.append(safety_labels)

        del model_inputs, output_ids
        torch.cuda.empty_cache()
        gc.collect()

    save_prompt_response_csv(
        PROMPTS,
        DAN_RESPONSES,
        DAN_SAFETY_LABELS,
        args.dir + "/mulit_generation_DAN.csv"
    )



    




