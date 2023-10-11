
import os 
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import os

from typing import Optional, Union

import torch
import torch.nn as nn

from peft import LoraConfig, get_peft_model 
from peft import PeftModel, PeftConfig

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig

from datasets import load_dataset
from datasets import Dataset
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax


warnings.filterwarnings("ignore", category=UserWarning)


# MODEL = 'meta-llama/Llama-2-7b-hf'
# MODEL = 'mistralai/Mistral-7B-v0.1'
MODEL = 'sharded_mistral_7b'
model_path = MODEL

model_type =  'multic'



## MODEL
model = AutoModelForCausalLM.from_pretrained(
    f'models/mistral_c/{model_path}',
    torch_dtype=torch.float16,
    trust_remote_code=True,
#     low_cpu_mem_usage=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(f'models/mistral_c/{model_path}')

tokenizer.pad_token = tokenizer.eos_token


# peft_model_id = "output/baseline-masterdata-llama2-lora-f32-sft"
peft_model_id = "output/baseline_mistral_c_all45k"
config = PeftConfig.from_pretrained(peft_model_id)


lora_model = PeftModel.from_pretrained(model, peft_model_id)

def get_ans(text):
    inputs = tokenizer(text, return_tensors='pt')
    logits = lora_model(input_ids=inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda()).logits[0, -1]
    
    # Create a list of tuples having (logit, 'option') format
     
#     options_list = [(logits[tokenizer(' A').input_ids[-1]].detach().cpu(), 'A'), (logits[tokenizer(' B').input_ids[-1]].detach().cpu(), 'B'), (logits[tokenizer(' C').input_ids[-1]].detach().cpu(), 'C'), (logits[tokenizer(' D').input_ids[-1]].detach().cpu(), 'D'), (logits[tokenizer(' E').input_ids[-1]].detach().cpu(), 'E')] 
    
    
    options_list = [(logits[tokenizer(' A').input_ids[-1]].detach().cpu(), 'A'), (logits[tokenizer(' B').input_ids[-1]].detach().cpu(), 'B'), (logits[tokenizer(' C').input_ids[-1]].detach().cpu(), 'C'), (logits[tokenizer(' D').input_ids[-1]].detach().cpu(), 'D')] 
    
    
    # Extracting the logits from the options_list
    logits = torch.tensor([option[0] for option in options_list])

    # Applying the softmax function to convert logits to probabilities
    probabilities = softmax(logits, dim=0)

    # Creating a new list with probabilities and corresponding options
    options_with_probabilities = [(prob.numpy(), option[1]) for prob, option in zip(probabilities, options_list)]

    options_list = sorted(options_list, reverse=True)
    ans_list = []
    for i in range(3):
        ans_list.append(options_list[i][1])
        
    return ans_list, options_with_probabilities


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.

    This function computes the average prescision at k between two lists of
    items.

    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements

    Returns
    -------
    score : double
            The average precision at k over the input lists

    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


## DATASET

MAX_CONTEXT = 1958
df = pd.read_csv('input_data/validation_data/master_validation_data_article_context.csv')
df = pd.read_csv('input_data/RACE_with_context_original.csv')
df['id'] = range(len(df))
# df['instruction'] = df['prompt'] + ' A: ' + df['A'] + ' B: ' + df['B'] + ' C: ' + df['C'] + ' D: ' + df['D'] + ' E: ' + df['E']
df['instruction'] = df['prompt'] + ' A: ' + df['A'] + ' B: ' + df['B'] + ' C: ' + df['C'] + ' D: ' + df['D'] 

# data_test = Dataset.from_pandas(df[['id','context','instruction','prompt','A','B','C','D','E']])
# system_msg = """Imagine three different experts are answering this question.
#  All experts will write down 1 step of their thinking, then share it with the group.
#  Then all experts will go on to the next step, etc.
#  If any expert realises they're wrong at any point then they leave.
#  The Question will be provided, and you have to answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D, E]"""


data_test = Dataset.from_pandas(df[['id','context','instruction','prompt','A','B','C','D']])
system_msg = """Imagine three different experts are answering this question.
 All experts will write down 1 step of their thinking, then share it with the group.
 Then all experts will go on to the next step, etc.
 If any expert realises they're wrong at any point then they leave.
 The Question will be provided, and you have to answer the following multiple choice question by giving the most appropriate response. Answer should be one among [A, B, C, D]"""

final_pred=[]
pred_op = tqdm(enumerate(data_test), total=len(data_test))
for i, data in pred_op:
    # text = f"### Question: {data['instruction']}\n ### Answer:"
    text = f"### Context:{data['context'][:MAX_CONTEXT]}\n\n ### Question: {data['instruction']}\n ### Answer:"
    ans_list, options_list = get_ans(text)
    final_pred.append({'id': data['id'],
                       'prediction':ans_list,
                        'logits':options_list})
    

submission_df = pd.DataFrame(final_pred)
submission_df['prediction'] = submission_df['prediction'].apply( lambda x: " ".join(x))
df['prediction'] = submission_df['prediction']
df['logits'] = submission_df['logits']




# https://www.kaggle.com/code/philippsinger/h2ogpt-perplexity-ranking
import numpy as np
def precision_at_k(r, k):
    """Precision at k"""
    assert k <= len(r)
    assert k != 0
    return sum(int(x) for x in r[:k]) / k

def MAP_at_3(predictions, true_items):
    """Score is mean average precision at 3"""
    U = len(predictions)
    map_at_3 = 0.0
    for u in range(U):
        user_preds = predictions[u].split()
        user_true = true_items[u]
        user_results = [1 if item == user_true else 0 for item in user_preds]
        for k in range(min(len(user_preds), 3)):
            map_at_3 += precision_at_k(user_results, k+1) * user_results[k]
    return map_at_3 / U

df.to_csv('base/validation_data_mistral_LORA_predicted_racedataset.csv',index=False)
m = MAP_at_3(df.prediction.values, df.answer.values)
print( 'CV MAP@3 =',m )