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
from peft import LoraConfig
parser = argparse.ArgumentParser(description='Generation Arguments')
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--dataset", type=str, default="Pankayaraj/STAR-41K-DA-Filtered-DeepSeek-R1-Distill-Qwen-32B")
parser.add_argument("--hf_token", type=str, default="")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=3)
parser.add_argument("--save_dir", type=str, default="path/to/sft/")

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

prompt_template, completion_template = get_chat_template(args.model)
def preprocess(data, prompt_template, completion_template):
    return {
        "prompt": prompt_template.format(prompt=data["question"]),
        "completion": completion_template.format(response=data["response"]),

        # "text": template.format(
        #     prompt=data["question"],
        #     response=data["response"]
        # )
    }



train_dts = load_dataset(args.dataset)["train"]
train_dts_processed = train_dts.map(
    preprocess,
    remove_columns=["question", "response", "category", "prompt"],
    fn_kwargs = {"prompt_template":prompt_template, "completion_template":completion_template},
    load_from_cache_file=False,
)
save_dir = args.save_dir

model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", dtype=torch.bfloat16, token=args.hf_token)
tokenizer = AutoTokenizer.from_pretrained(args.model, token=args.hf_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

if "DeepSeek" in args.dataset and "DeepSeek" not in args.model:
    special_tokens_dict = {'additional_special_tokens': ['<think>', '</think>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    print(f"Added special tokens: {special_tokens_dict['additional_special_tokens']}")

training_args = SFTConfig(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        completion_only_loss=True,
        activation_offloading=False,
        num_train_epochs=args.epochs,
        output_dir=save_dir,
        logging_first_step=True,
        logging_steps=500, 
        learning_rate=1.41e-5,
        save_strategy="epoch",
        save_only_model=True,
        max_length=2048,
)


peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        modules_to_save=['lm_head'],
        task_type="CAUSAL_LM",
)


trainer = SFTTrainer(
    model,
    processing_class=tokenizer,
    train_dataset=train_dts_processed,
    args=training_args,
    peft_config=peft_config,
)
trainer.train()