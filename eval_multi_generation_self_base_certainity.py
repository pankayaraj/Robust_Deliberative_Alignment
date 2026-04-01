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


parser = argparse.ArgumentParser(description='Generation Arguments')
parser.add_argument("--dir", type=str, default="results/MODEL_Qwen2.5-1.5B-Instruct_DATASET_STAR-41K-DA-Filtered-DeepSeek-R1-Distill-Qwen-32B/3")
parser.add_argument("--hf_token", type=str, default="")
parser.add_argument("--batch_size", type=int, default=2)

args = parser.parse_args()

import csv
def save_prompt_response_csv(list_a, list_b, list_c, list_d, list_e, list_f, filename, col1="prompt", col2="response", col3="label", col4="kl", col5="reverser_kl", col6="perplexity"):

    seperator = "__SEPERATOR__"
    list_b = [seperator.join(l_b) for l_b in list_b]
    list_c = [seperator.join(l_c) for l_c in list_c]
    list_d = [seperator.join(l_d) for l_d in list_d]
    list_e = [seperator.join(l_e) for l_e in list_e]
    list_f = [seperator.join(l_f) for l_f in list_f]

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([col1, col2, col3, col4, col5, col6])   # header
        for a, b, c, d, e, f in zip(list_a, list_b, list_c, list_d, list_e, list_f):
            writer.writerow([a, b, c, d, e, f])

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


def compute_perplexity(inputs, model, prompt_length):
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.float() 

    # 2. Shift logits and labels for next-token prediction
    # Logits: [batch_size, seq_len-1, vocab_size]
    shift_logits = logits[:, :-1, :].contiguous()
    # Labels: [batch_size, seq_len-1]
    shift_labels = inputs["input_ids"][:, 1:].contiguous()
    # 3. Compute Cross-Entropy Loss per token (no reduction)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(shift_labels.size()) # Reshape back to [batch_size, seq_len-1]

    # 4. Mask out padding tokens so they don't affect the score

    attention_mask = inputs["attention_mask"]
    attention_mask[:,:prompt_length] = torch.zeros(inputs["attention_mask"].shape[0], prompt_length)
    mask = attention_mask[:, 1:].contiguous()
    masked_loss = loss * mask

    # 5. Calculate Perplexity per sample
    # Sum of log-likelihoods divided by number of non-padding tokens
    sum_loss = masked_loss.sum(dim=1)
    count_tokens = mask.sum(dim=1)
    avg_loss = sum_loss / count_tokens

   
    per_sample_negative_perplexity = torch.exp(avg_loss).squeeze()

    result = per_sample_negative_perplexity.cpu().tolist()
    if type(result) == float:
        result = [result]
    return result

def compute_reverse_KL(inputs, model, base_model, prompt_length):
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.float()

        base_outputs = base_model(**inputs)
        base_logits = base_outputs.logits.float()

    # logits_probs = F.softmax(logits, dim=-1)
    # base_log_probs = F.log_softmax(base_logits, dim=-1)
    # kl_pointwise = F.kl_div(
    #     input=base_log_probs, 
    #     target=logits_probs, 
    #     reduction='none'
    # )
    # kl_per_token = kl_pointwise.sum(dim=-1)

    log_p = F.log_softmax(logits, dim=-1)
    log_q = F.log_softmax(base_logits, dim=-1)
    q = torch.exp(log_q)
    kl_elementwise = q * (log_q - log_p)
    kl_per_token = torch.sum(kl_elementwise, dim=-1)

    attention_mask = inputs["attention_mask"]
    attention_mask[:,:prompt_length] = torch.zeros(inputs["attention_mask"].shape[0], prompt_length)
    mask = attention_mask[:, :].contiguous()
    masked_kl = kl_per_token * mask

    sum_kl = masked_kl.sum(dim=1)
    count_tokens = mask.sum(dim=1)
    avg_loss = sum_kl / count_tokens

    result = avg_loss.cpu().tolist()
    if type(result) == float:
        result = [result]

    del logits, base_logits, outputs, base_outputs, log_p, log_q, q, kl_elementwise
    torch.cuda.empty_cache()
    gc.collect()

    # del logits, base_logits, kl_pointwise
    # torch.cuda.empty_cache()
    # gc.collect()

    return result

def compute_KL(inputs, model, base_model, prompt_length):
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.float()

        base_outputs = base_model(**inputs)
        base_logits = base_outputs.logits.float()

    log_p = F.log_softmax(logits, dim=-1)
    log_q = F.log_softmax(base_logits, dim=-1)
    p = torch.exp(log_p)
    kl_elementwise = p * (log_p - log_q)
    kl_per_token = torch.sum(kl_elementwise, dim=-1)

    # logits_log_probs = F.log_softmax(logits, dim=-1)
    # base_probs = F.softmax(base_logits, dim=-1)
    # kl_pointwise = F.kl_div(
    #     input=logits_log_probs, 
    #     target=base_probs, 
    #     reduction='none'
    # )
    # kl_per_token = kl_pointwise.sum(dim=-1)

    attention_mask = inputs["attention_mask"]
    attention_mask[:,:prompt_length] = torch.zeros(inputs["attention_mask"].shape[0], prompt_length)
    mask = attention_mask[:, :].contiguous()
    masked_kl = kl_per_token * mask

    sum_kl = masked_kl.sum(dim=1)
    count_tokens = mask.sum(dim=1)
    avg_loss = sum_kl / count_tokens

    result = avg_loss.cpu().tolist()
    if type(result) == float:
        result = [result]

    del logits, base_logits, outputs, base_outputs, log_p, log_q, p, kl_elementwise
    torch.cuda.empty_cache()
    gc.collect()

    # del logits, base_logits, kl_pointwise
    # torch.cuda.empty_cache()
    # gc.collect()

    return result


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

 



# DAN = pd.read_csv(args.dir + "/rejected_mulit_generation_DAN.csv")
# Strong_Reject = pd.read_csv(args.dir + "/rejected_mulit_generation_strongreject.csv")



DATASETS = [0,0,1]
batch_size = 1 
"""
    Always keep it to 1 batch size
    Keep padding to right
"""

PROMPT_TEMPLATE, COMPLETION_TEMPLATE = get_chat_template(config.base_model_name_or_path)
seperator = "__SEPERATOR__"


if DATASETS[0] == 1:
    Wildjailbreak = pd.read_csv(args.dir + "/mulit_generation_wildjailbreak.csv")
    PROMPTS = Wildjailbreak["prompt"].tolist()
    RESPONSES = [r.split(seperator) for r in Wildjailbreak["response"].tolist()]
    LABELS_STR = [l.split(seperator) for l in Wildjailbreak["label"].tolist()]
    LABELS = [list(map(int, l.split(seperator)))  for l in Wildjailbreak["label"].tolist()]
    KL = []
    REVERSE_KL = []
    PERPLEXITY = []
    
    
    num_generations = len(RESPONSES[0])

    
    for idx in tqdm(range(0, len(PROMPTS))):

        prompt = PROMPTS[idx]
        responses = RESPONSES[idx]
        labels = LABELS[idx]
        
        # for r in responses:
        #     if r == "":
        #         responses = "       "

        # if responses == "":
        #     responses = "       "

        question = PROMPT_TEMPLATE.format(prompt=prompt) + "<think>"
        full_pairs = [question + responses[i] for i in range(len(responses))]

        
        prompt_ids = tokenizer([question], return_tensors="pt", max_length=2048, padding=True, truncation=True, padding_side="right")
        prompt_length = prompt_ids["input_ids"].shape[1]

        
        # attention_indices = full_ids["attention_mask"].sum(1).tolist()
        # total_shape = full_ids["input_ids"].shape[1]
        # attention_indices = [total_shape - i for i in attention_indices]

        kl = []
        
        incr = 4
        for i in range(0,8,incr):   
            full_ids = tokenizer(full_pairs[i:i+incr], return_tensors="pt", max_length=2048, padding=True, truncation=True, padding_side="right")
            kl += compute_KL(full_ids, model, base_model, prompt_length)
        kl = [str(k) for k in kl]
        KL.append(kl)

        reverse_kl = []
        for i in range(0,8,incr):   
            full_ids = tokenizer(full_pairs[i:i+incr], return_tensors="pt", max_length=2048, padding=True, truncation=True, padding_side="right")
            reverse_kl += compute_reverse_KL(full_ids, model, base_model, prompt_length)
        reverse_kl = [str(k) for k in reverse_kl]
        REVERSE_KL.append(reverse_kl)

        perplexity = []
        for i in range(0,8,incr):   
            full_ids = tokenizer(full_pairs[i:i+incr], return_tensors="pt", max_length=2048, padding=True, truncation=True, padding_side="right")
            perplexity += compute_perplexity(full_ids, base_model, prompt_length)
        perplexity = [str(k) for k in perplexity]
        PERPLEXITY.append(perplexity)

        

        torch.cuda.empty_cache()
        gc.collect()

        

    save_prompt_response_csv(
        PROMPTS,
        RESPONSES,
        LABELS_STR,
        KL,
        REVERSE_KL,
        PERPLEXITY,
        args.dir + "/mulit_generation_self_base_certainity_wildjailbreak.csv"
    )



if DATASETS[1] == 1:
    StrongReject = pd.read_csv(args.dir + "/mulit_generation_strongreject.csv")
    PROMPTS = StrongReject["prompt"].tolist()
    RESPONSES = [r.split(seperator) for r in StrongReject["response"].tolist()]
    LABELS_STR = [l.split(seperator) for l in StrongReject["label"].tolist()]
    LABELS = [list(map(int, l.split(seperator)))  for l in StrongReject["label"].tolist()]
    KL = []
    REVERSE_KL = []
    PERPLEXITY = []
    
    
    num_generations = len(RESPONSES[0])

    
    for idx in tqdm(range(0, len(PROMPTS))):

        prompt = PROMPTS[idx]
        responses = RESPONSES[idx]
        labels = LABELS[idx]
        
        # for r in responses:
        #     if r == "":
        #         responses = "       "

        # if responses == "":
        #     responses = "       "

        question = PROMPT_TEMPLATE.format(prompt=prompt) + "<think>"
        full_pairs = [question + responses[i] for i in range(len(responses))]

        
        prompt_ids = tokenizer([question], return_tensors="pt", max_length=2048, padding=True, truncation=True, padding_side="right")
        prompt_length = prompt_ids["input_ids"].shape[1]

        
        # attention_indices = full_ids["attention_mask"].sum(1).tolist()
        # total_shape = full_ids["input_ids"].shape[1]
        # attention_indices = [total_shape - i for i in attention_indices]

        kl = []
        
        incr = 4
        for i in range(0,8,incr):   
            full_ids = tokenizer(full_pairs[i:i+incr], return_tensors="pt", max_length=2048, padding=True, truncation=True, padding_side="right")
            kl += compute_KL(full_ids, model, base_model, prompt_length)
        kl = [str(k) for k in kl]
        KL.append(kl)

        reverse_kl = []
        for i in range(0,8,incr):   
            full_ids = tokenizer(full_pairs[i:i+incr], return_tensors="pt", max_length=2048, padding=True, truncation=True, padding_side="right")
            reverse_kl += compute_reverse_KL(full_ids, model, base_model, prompt_length)
        reverse_kl = [str(k) for k in reverse_kl]
        REVERSE_KL.append(reverse_kl)

        perplexity = []
        for i in range(0,8,incr):   
            full_ids = tokenizer(full_pairs[i:i+incr], return_tensors="pt", max_length=2048, padding=True, truncation=True, padding_side="right")
            perplexity += compute_perplexity(full_ids, base_model, prompt_length)
        perplexity = [str(k) for k in perplexity]
        PERPLEXITY.append(perplexity)

        

        torch.cuda.empty_cache()
        gc.collect()

        

    save_prompt_response_csv(
        PROMPTS,
        RESPONSES,
        LABELS_STR,
        KL,
        REVERSE_KL,
        PERPLEXITY,
        args.dir + "/mulit_generation_self_base_certainity_strongreject.csv"
    )


if DATASETS[2] == 1:
    DAN = pd.read_csv(args.dir + "/mulit_generation_DAN.csv")
    PROMPTS = DAN["prompt"].tolist()
    RESPONSES = [r.split(seperator) for r in DAN["response"].tolist()]
    LABELS_STR = [l.split(seperator) for l in DAN["label"].tolist()]
    LABELS = [list(map(int, l.split(seperator)))  for l in DAN["label"].tolist()]
    KL = []
    REVERSE_KL = []
    PERPLEXITY = []
    
    
    num_generations = len(RESPONSES[0])
    
    for idx in tqdm(range(0, len(PROMPTS))):

        prompt = PROMPTS[idx]
        responses = RESPONSES[idx]
        labels = LABELS[idx]

        # if responses == "":
        #     responses = "       "

        # for r in responses:
        #     if r == "":
        #         responses[i] = "       "

        question = PROMPT_TEMPLATE.format(prompt=prompt) + "<think>"
        full_pairs = [question + responses[i] for i in range(len(responses))]

        
        prompt_ids = tokenizer([question], return_tensors="pt", max_length=2048, padding=True, truncation=True, padding_side="right")
        prompt_length = prompt_ids["input_ids"].shape[1]

        
        # attention_indices = full_ids["attention_mask"].sum(1).tolist()
        # total_shape = full_ids["input_ids"].shape[1]
        # attention_indices = [total_shape - i for i in attention_indices]

        kl = []
        
        
        incr = 4
        for i in range(0,8,incr):
            
        
            full_ids = tokenizer(full_pairs[i:i+incr], return_tensors="pt", max_length=2048, padding=True, truncation=True, padding_side="right")
            kl += compute_KL(full_ids, model, base_model, prompt_length)
        kl = [str(k) for k in kl]
        KL.append(kl)

        reverse_kl = []
        for i in range(0,8,incr):   
            full_ids = tokenizer(full_pairs[i:i+incr], return_tensors="pt", max_length=2048, padding=True, truncation=True, padding_side="right")
            reverse_kl += compute_reverse_KL(full_ids, model, base_model, prompt_length)
        reverse_kl = [str(k) for k in reverse_kl]
        REVERSE_KL.append(reverse_kl)

        perplexity = []
        for i in range(0,8,incr):   
            full_ids = tokenizer(full_pairs[i:i+incr], return_tensors="pt", max_length=2048, padding=True, truncation=True, padding_side="right")
            perplexity += compute_perplexity(full_ids, base_model, prompt_length)
        perplexity = [str(k) for k in perplexity]
        PERPLEXITY.append(perplexity)

        

        torch.cuda.empty_cache()
        gc.collect()

        

    save_prompt_response_csv(
        PROMPTS,
        RESPONSES,
        LABELS_STR,
        KL,
        REVERSE_KL,
        PERPLEXITY,
        args.dir + "/mulit_generation_self_base_certainity_DAN.csv"
    )