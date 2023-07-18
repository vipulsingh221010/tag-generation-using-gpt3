import numpy as np
import torch
import av
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
import csv
import random
import shutil
import time
from huggingface_hub import hf_hub_download
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
model1 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device)

with open('tags_gpt35_ucf.json') as file:
    data=json.load(file)
similarity_scores=[]
video_scores={}
label=[]
for key in data:

    if data[key]['truth_label'] not in label:
        label.append(data[key]['truth_label'])
embeddings={}
for val in label:
    embedding_1 = model1.encode(val, convert_to_tensor=True).to(device)
    embeddings[val]=embedding_1
precision_list=[]
recall_list=[]
f1_list=[]
recall_list=[]
accuracy_list=[]

for key in data:
    t1=time.time()
    truth_label=data[key]['truth_label']
    tags=data[key]['tags']
    embedding2={}
    embedding_1=embeddings[truth_label]
    scores=[]
    for tag in tags:
        embedding2[tag]=model1.encode(tag, convert_to_tensor=True).to(device)
        embedding_2=embedding2[tag]
        score=util.pytorch_cos_sim(embedding_1, embedding_2).to(device)
        score=score.cpu().item()
        scores.append(score)
        #print(sim_max)
    truth_labels=np.ones(len(scores))
    precision = precision_score(truth_labels, [int(score >= 0.4) for score in scores])
    recall = recall_score(truth_labels, [int(score >= 0.4) for score in scores])
    f1 = f1_score(truth_labels, [int(score >= 0.4) for score in scores])
    accuracy=accuracy_score(truth_labels, [int(score >= 0.4) for score in scores])
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    accuracy_list.append(accuracy)
    video_scores[key]=scores
    
    t2=time.time()
    print(t2-t1)

precision = sum(precision_list) / len(precision_list)

f1= sum(f1_list) / len(f1_list)
recall = sum(recall_list) / len(recall_list)
accuracy= sum(accuracy_list) / len(accuracy_list)
print(precision)
print(accuracy)
print(f1)
print(recall)

data=list(zip(precision_list,recall_list,f1_list,accuracy_list))
csv_file='results_gpt_ucf.csv'

with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Prescision', 'Recall', 'F1-score','Accuracy'])  # Write the header row
    writer.writerows(data)  # Write the data rows


file_path='Sample_ucf_scores_gpt.csv'


with open(file_path,mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Key', 'Value'])  # Write the header row

    for key, values in video_scores.items():
        writer.writerow([key] + values)
