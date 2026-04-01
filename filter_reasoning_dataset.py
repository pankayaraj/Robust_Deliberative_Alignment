from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
import argparse
from tqdm import tqdm
import torch
import gc
import numpy as np
import re


parser = argparse.ArgumentParser(description='Generation Arguments')
parser.add_argument("--dataset", type=str, default="Pankayaraj/STAR-41K-DA-Unfiltered-DeepSeek-R1-Distill-Llama-8B")
parser.add_argument("--hf_token", type=str, default="")
parser.add_argument("--batch_size", type=int, default=32)

args = parser.parse_args()

dts = load_dataset(args.dataset)["train"]
QUESTIONS = dts["question"]
RESPONSES = dts["response"]

total_size = len(dts)


SAFETY_LABEL = []
REFUSAL_LABEL = []


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "meta-llama/Llama-Guard-3-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=args.hf_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto", token=args.hf_token)



for idx in tqdm(range(0, total_size, args.batch_size)):
    
    start = idx
    end = min(idx + args.batch_size, total_size)

    prompt = QUESTIONS[start:end]
    response = RESPONSES[start:end]

    
    text = [
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": p},
                {"role": "assistant", "content": r},
            ],
            tokenize=False
        ) for p, r in zip(prompt,response)
    ]


    model_inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(model.device)
    output_ids = model.generate(
            **model_inputs,
            max_new_tokens=100,
    )

    # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    output_ids = output_ids[:, len(model_inputs.input_ids[0]):]
    content = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    
    safety_labels = [1 if c.strip().lower().startswith("safe") else 0 for c in content] 

    SAFETY_LABEL += safety_labels
    
    del output_ids, model_inputs
    torch.cuda.empty_cache()
    gc.collect()
    
  


question_harmful = []
question_harmless = []

category_harmful = []
category_harmless = []

prompt_harmful = []
prompt_harmless = []

response_harmful = []
response_harmless = []

for l, d in tqdm(zip(SAFETY_LABEL, dts)):

    if l == 1:
        question_harmless.append(d["question"])
        category_harmless.append(d["category"])
        prompt_harmless.append(d["prompt"])
        response_harmless.append(d["response"])
    else:
        question_harmful.append(d["question"])
        category_harmful.append(d["category"])
        prompt_harmful.append(d["prompt"])
        response_harmful.append(d["response"])


from datasets import Dataset, DatasetDict
train = {
    "question": question_harmless,
    "category": category_harmless,
    "prompt": prompt_harmless,
    "response": response_harmless
}

rejected = {
    "question": question_harmful,
    "category": category_harmful,
    "prompt": prompt_harmful,
    "response": response_harmful
}
dataset = DatasetDict({
    "train": Dataset.from_dict(train),
    "filtered": Dataset.from_dict(rejected),
})


dts_name = args.dataset.replace("Unfiltered", "Filtered")
dataset.push_to_hub(dts_name, token=args.hf_token)


