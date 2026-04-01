from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
import gc
import numpy as np
from trl import SFTConfig, SFTTrainer
import wandb
import copy

from peft import LoraConfig,PeftConfig, PeftModel,inject_adapter_in_model, get_peft_model
import pandas as pd
import torch
import torch.nn.functional as F


parser = argparse.ArgumentParser(description='Generation Arguments')
parser.add_argument("--dir", type=str, default="results/MODEL_Qwen2.5-1.5B-Instruct_DATASET_STAR-41K-DA-Filtered-DeepSeek-R1-Distill-Qwen-32B/3")
parser.add_argument("--hf_token", type=str, default="")
parser.add_argument("--batch_size", type=int, default=2)

args = parser.parse_args()

import csv
def save_prompt_response_csv(list_a, list_b, list_c, filename, col1="prompt", col2="response", col3="embedding_similairty"):

    seperator = "__SEPERATOR__"
    list_b = [seperator.join(l_b) for l_b in list_b]
    list_c = [seperator.join(l_c) for l_c in list_c]


    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([col1, col2, col3])   # header
        for a, b, c in zip(list_a, list_b, list_c):
            writer.writerow([a, b, c])

def get_chat_template(model_name):
    if "qwen" in model_name.lower():
        PROMPT_TEMPLATE = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\nAnswer: Let's think step by step."
        COMPLETION_TEMPLATE = "<think>{response}<|im_end|>\n"
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant"
    elif "llama" in model_name.lower():
        PROMPT_TEMPLATE = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAnswer: Let's think step by step."
        COMPLETION_TEMPLATE = "<think>{response}<|eot_id|>"
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>"
    elif "gemma" in model_name.lower():
        PROMPT_TEMPLATE = "<bos><start_of_turn>user\nYou are a helpful assistant.\n\n{prompt}<end_of_turn><start_of_turn>model\nAnswer: Let's think step by step."
        COMPLETION_TEMPLATE = "<think>{response}<end_of_turn>\n"
        instruction_template = "<start_of_turn>user"
        response_template = "<start_of_turn>model"
    
    return PROMPT_TEMPLATE, COMPLETION_TEMPLATE




def compute_embedding_similarity(inputs, model, base_model, prompt_length):
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states

        base_outputs = base_model(**inputs, output_hidden_states=True)
        hs_base = base_outputs.hidden_states
    
    batch_size = hs_base[0].shape[0]
    layer_similarities = [[] for _ in range(batch_size)]

    # For each layer
    for i in range(len(hs)):
        
        last_token_hs = hs[i][:, -1, :]
        last_token_hs_base = hs_base[i][:, -1, :]
        
        sim = F.cosine_similarity(last_token_hs, last_token_hs_base, dim=-1)
        similairty = sim.squeeze().cpu().tolist()
        for k in range(batch_size):
            layer_similarities[k].append(str(similairty[k]))
        
    
    layer_similarities = ["_".join(ls) for ls in layer_similarities]


    del outputs, base_outputs, hs_base, hs, sim
    torch.cuda.empty_cache()
    gc.collect()
    
    return layer_similarities

    



model_dir = args.dir.replace("/results/", "/output/")
tokenizer = AutoTokenizer.from_pretrained(model_dir, token=args.hf_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

config = PeftConfig.from_pretrained(model_dir)

model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, device_map="auto", dtype=torch.bfloat16, token=args.hf_token)
model.config.use_cache = False
if "DeepSeek" in model_dir and "DeepSeek" not in config.base_model_name_or_path:
    model.resize_token_embeddings(len(tokenizer))



base_model = copy.deepcopy(model)

model = PeftModel.from_pretrained(model, model_dir)
model = model.merge_and_unload()

 

"""
    Always keep it to 1 batch size
    Keep padding to right
"""

PROMPT_TEMPLATE, COMPLETION_TEMPLATE = get_chat_template(config.base_model_name_or_path)
seperator = "__SEPERATOR__"


mmlu = pd.read_csv(args.dir + "/mulit_generation_MMLU.csv")
PROMPTS = mmlu["prompt"].tolist()

#this is just to make up for the error of saving the columns wrongly. I saved responses in "category" column
RESPONSES = [r.split(seperator) for r in mmlu["category"].tolist()] #[r.split(seperator) for r in mmlu["response"].tolist()]
EMBEDDING_SIMILAIRITY = []
    
    
num_generations = len(RESPONSES[0])

    
for idx in tqdm(range(0, len(PROMPTS))):

    prompt = PROMPTS[idx]
    responses = RESPONSES[idx]


    question = PROMPT_TEMPLATE.format(prompt=prompt)
    full_pairs = [question + responses[i] for i in range(len(responses))]

    
        
    prompt_ids = tokenizer([question], return_tensors="pt", max_length=2048, padding=True, truncation=True, padding_side="right")
    prompt_length = prompt_ids["input_ids"].shape[1]


    embedding_similairty = []
    incr = 4
    for i in range(0,8,incr):   
        full_ids = tokenizer(full_pairs[i:i+incr], return_tensors="pt", max_length=2048, padding=True, truncation=True, padding_side="right")
        embedding_similairty += compute_embedding_similarity(full_ids, model, base_model, prompt_length)
    EMBEDDING_SIMILAIRITY.append(embedding_similairty)        


        

    torch.cuda.empty_cache()
    gc.collect()

        

save_prompt_response_csv(
    PROMPTS,
    RESPONSES,
    EMBEDDING_SIMILAIRITY,
    args.dir + "/mulit_generation_self_base_embedding_similarity_MMLU.csv"
)
