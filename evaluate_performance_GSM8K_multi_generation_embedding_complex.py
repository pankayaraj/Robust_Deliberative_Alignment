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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm

import random
random.seed(42)

import argparse
parser = argparse.ArgumentParser(description='Generation Arguments')
parser.add_argument("--dir", type=str, default="results/MODEL_Qwen2.5-1.5B-Instruct_DATASET_STAR-41K-DA-Filtered-DeepSeek-R1-Distill-Qwen-32B/3")
args = parser.parse_args()


datasets = ["GSM8K"]
dataset_names = ["GSM8K"]

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
def extract_predicted_answer(text, ground_truth):
    regex_pattern = "(-?[$0-9.,]{2,})|(-?[0-9]+)"
    regexes_to_ignore =[
        ",",
        "\\$",
        "(?s).*#### ",
        "\\.$"
    ]
    match = re.findall(regex_pattern, text)
    if match:
        match = match[-1]
        if isinstance(match, tuple):
            match = [m for m in match if m][0]
        text = match.strip()

        for regex in regexes_to_ignore:
            text = re.sub(regex, "", text)

        if text == ground_truth or ground_truth in text:
            return 1
        else:
            return 0
    else:
        return 0

for i, d in enumerate(datasets):
        
        

        
    cat = "self_base_embedding_similarity"
    metrics = categories_map[cat]

    
    load_dir = args.dir + "/mulit_generation_" + cat + "_" + str(d) + ".csv"
    load_dir_base = args.dir + "/mulit_generation_" + str(d) + ".csv"

    responses = pd.read_csv(load_dir)["response"].tolist()
    response_list = [list(map(str, r.split(seperator))) for r in responses]

    ground_truth_labels = pd.read_csv(load_dir_base)["golden_answer"]
    ground_truth_labels = [l for l in ground_truth_labels]

    label_list = [[extract_predicted_answer(r, ground_truth_labels[k]) for r in res] for k, res in enumerate(response_list)]

            
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

        