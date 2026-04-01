from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
import argparse
from tqdm import tqdm
import torch
import gc
import numpy as np
from trl import GRPOConfig, GRPOTrainer
import wandb
from peft import LoraConfig
from torch import Tensor

from peft import LoraConfig,PeftConfig, PeftModel,inject_adapter_in_model, get_peft_model

parser = argparse.ArgumentParser(description='Generation Arguments')
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
parser.add_argument("--dataset", type=str, default="Pankayaraj/STAR-41K-DA-Filtered-DeepSeek-R1-Distill-Qwen-32B")
parser.add_argument("--hf_token", type=str, default="")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--accumulation_size", type=int, default=8)
parser.add_argument("--epochs", type=int, default=1)
parser.add_argument("--sft_epoch", type=int, default=3)
parser.add_argument("--reward_batch_size", type=int, default=32)

parser.add_argument("--save_dir", type=str, default="path/to/sft/")
parser.add_argument("--sft_dir", type=str, default="path/to/grpo/")

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
        "prompt_original": data["question"],
    }


train_dts = load_dataset(args.dataset)["train"]
train_dts_processed = train_dts.map(
    preprocess,
    remove_columns=["question", "response", "category", "prompt"],
    fn_kwargs = {"prompt_template":prompt_template, "completion_template":completion_template},
    load_from_cache_file=False,
)

filter_model_id = "meta-llama/Llama-Guard-3-8B"
filter_tokenizer = AutoTokenizer.from_pretrained(filter_model_id , token=args.hf_token)
if filter_tokenizer.pad_token is None:
    filter_tokenizer.pad_token = filter_tokenizer.eos_token
filter_model = AutoModelForCausalLM.from_pretrained(filter_model_id, torch_dtype=torch.bfloat16, device_map="balanced_low_0", token=args.hf_token)

def reward_func(
    prompts: list[str],
    completions: list[str],
    completion_ids: list[list[int]] | list[Tensor],
    trainer_state,
    prompt_original: list[str],
    **kwargs
):

    rewards = []  
    bs = args.reward_batch_size
    for idx in range(0, len(prompt_original), bs):
        start = idx
        end = min(idx + bs, len(prompt_original))

        question = prompt_original[start:end]
        response = completions[start:end]

        text = [
            filter_tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": p},
                    {"role": "assistant", "content": r},
                ],
                tokenize=False
            ) for p, r in zip(question,response)
        ]

        model_inputs = filter_tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(filter_model.device)
        output_ids = filter_model.generate(
                **model_inputs,
                max_new_tokens=100,
        )

        output_ids = output_ids[:, len(model_inputs.input_ids[0]):]
        content = filter_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        
        safety_labels = [1.0 if c.strip().lower().startswith("safe") else 0.0 for c in content] 
        rewards += safety_labels

        del output_ids, model_inputs
        torch.cuda.empty_cache()
        gc.collect()

    return rewards


model_dir = args.sft_dir
save_dir = args.save_dir
tokenizer = AutoTokenizer.from_pretrained(model_dir, token=args.hf_token)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", dtype=torch.bfloat16, token=args.hf_token)
model.config.use_cache = False
model.resize_token_embeddings(len(tokenizer))
config = PeftConfig.from_pretrained(model_dir)
model = PeftModel.from_pretrained(model, model_dir)


training_args = GRPOConfig(
    beta=0.0,
    per_device_train_batch_size=args.batch_size,
    gradient_accumulation_steps=args.accumulation_size,
    num_train_epochs=args.epochs,
    output_dir=save_dir,
    logging_first_step=True,
    logging_steps=50, 
    learning_rate=1.41e-5,
    save_strategy="steps",
    save_steps=500,
    save_only_model=True,
    max_completion_length=2048,
)

trainer = GRPOTrainer(
    model=model,
    args=training_args, 
    processing_class=tokenizer, 
    reward_funcs=reward_func,
    train_dataset=train_dts_processed,
)

trainer.train()



