from scipy.stats import gaussian_kde

import json
import os
from plotly import graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import plotly.io as pio   
pio.kaleido.scope.mathjax = None
import csv
import pandas as pd
import math


import numpy as np
import torch
import numpy as np
from tqdm import tqdm

import random
random.seed(42)



import argparse
parser = argparse.ArgumentParser(description='Generation Arguments')
parser.add_argument("--dir", type=str, default="results/MODEL_Qwen2.5-1.5B-Instruct_DATASET_STAR-41K-DA-Filtered-DeepSeek-R1-Distill-Qwen-32B/3")
args = parser.parse_args()


datasets = ["MMLU"]
dataset_names = ["MMLU"]

epoch = 3
seperator = "__SEPERATOR__"



metrics_categories = ["self_base_embedding_similarity"]

categories_map = {
    "self_base_embedding_similarity":["embedding_similairty", ],
}

axis_map = {
    "embedding_similairty": "Embedding cosine similairty"
}

func_map = {
    "self_base_embedding_similarity":{
        "embedding_similairty":np.argmin,
    },
}

import re


def extract_answer(text):
    pattern = r"answer is \(?([A-J])\)?"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return extract_again(text)


def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text)
    if match:
        return match.group(1)
    else:
        return extract_final(text)


def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

from datasets import load_dataset
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

CHOICES = []
CHOICE_LABLES = []
choice_map = {
    0:"A",
    1:"B",
    2:"C",
    3:"D",
}
for name, mmlu in zip(MMLU_SPLITS, MMLU_LIST):
    for data in mmlu:
        answer = data["choices"][data["answer"]]
        CHOICES.append(answer)
        CHOICE_LABLES.append(data["answer"])
def extract_predicted_answer(text, answer, label):
    if answer in text:
        return 1

    e_text = extract_answer(text)
    if e_text is None:
        return 0
    else:
        if e_text == choice_map[label]:
            return 1
        else:
            return 0


for i, d in enumerate(datasets):
        

    cat = "self_base_embedding_similarity"
    metrics = categories_map[cat]

            
    load_dir = args.dir + "/mulit_generation_" + cat + "_" + str(d) + ".csv"
    load_dir_base = args.dir + "/mulit_generation_" + str(d) + ".csv"

    responses = pd.read_csv(load_dir)["response"].tolist()
    response_list = [list(map(str, r.split(seperator))) for r in responses]

    ground_truth_labels = CHOICE_LABLES
    ground_truth = CHOICES

    label_list = [[extract_predicted_answer(r, ground_truth[k], ground_truth_labels[k]) for r in res] for k, res in enumerate(response_list)]

            
    metrics_list = [
                [
                    list(map(str, l.split(seperator))) 
                    for l in pd.read_csv(load_dir)[m].tolist()
                ]             
                for m in metrics
    ]

    for s in range(len(metrics_list)):
        for t in range(len(metrics_list[0])):
            for u in range(len(metrics_list[0][0])):
                metrics_list[s][t][u] = list(map(float, metrics_list[s][t][u].split("_"))) 

    num_layers = len(metrics_list[0][0][0])

    safety_labels = [[] for _ in range(num_layers)]
    safety_labels_base = []
    safety_labels_best = []
    safety_labels_atleast_one = []
    safety_labels_random = []

    for s in range(len(metrics)):
        for v_l, l_l in zip(metrics_list[s], label_list): 
            for n_l in range(num_layers):
                val = [v_l[t][n_l] for t in range(len(v_l))]
                l_l = l_l

                        
                idx = np.argmin(val)
                safety_labels[n_l].append(l_l[idx])

            safety_labels_base.append(l_l[0])
            if sum(l_l) > 4:
                safety_labels_best.append(1)
            else:
                safety_labels_best.append(0)
                    
            if sum(l_l) > 1:
                safety_labels_atleast_one.append(1)
            else:
                safety_labels_atleast_one.append(0)
                    
            choice = random.randint(0,7)
            safety_labels_random.append(l_l[choice])
                    
        asr = [100*sum(sl)/len(sl) for sl in safety_labels]
        asr_base = 100*sum(safety_labels_base)/len(safety_labels_base)
        asr_best = 100*sum(safety_labels_best)/len(safety_labels_best)
        asr_atleast_one = 100*sum(safety_labels_atleast_one)/len(safety_labels_atleast_one)
        asr_random = 100*sum(safety_labels_random)/len(safety_labels_random)
                
        asr_dict = {
                    "single_sample": asr_base,
                    "majority_vote": asr_best,
                    "atleast_one": asr_atleast_one,
                    "random":asr_random,
                    "best_layer":max(asr),
                    "best_layer_no":float(np.argmax(asr)),
        }

        for t in range(num_layers):
            asr_dict["layer_" + str(t)] = asr[t]


                
        save_dir = args.dir + "/" 
        with open(save_dir + "performance_" + str(d) + ".json", "w") as f:
            json.dump(asr_dict, f, indent=4)