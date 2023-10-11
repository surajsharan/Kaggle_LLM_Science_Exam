
import os 
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb

from peft import LoraConfig, get_peft_model 
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from langchain.prompts import PromptTemplate
from transformers import TrainingArguments
from datasets import load_dataset
from datasets import Dataset


warnings.filterwarnings("ignore", category=UserWarning)



model_name = "mistralai/Mistral-7B-v0.1"
MAX_CONTEXT = 2500


def _get_model_tokeinzer(path, model_type):
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    if model_type == 'causal':
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            # low_cpu_mem_usage=True,
            device_map="auto"
        )
    elif model_type == 'seq':
        model = AutoModelForSequenceClassification.from_pretrained(path,num_labels=1)
    elif model_type == 'multic':
        model = AutoModelForMultipleChoice.from_pretrained(path)
    
    return tokenizer, model
    

model_type = 'causal'
directory = f"models/mistral_c/{model_name}"



if os.path.exists(directory):
    print(f"Loading Model from directory: '{directory}' ")
    tokenizer, model = _get_model_tokeinzer(directory,model_type)
else:
    print(f"Downloading Model from huggingface: '{model_name}' ")
    tokenizer, model = _get_model_tokeinzer(model_name,model_type)
    # Save the model weights to disk
    os.makedirs(directory, exist_ok=True)
    print(f"Saving the Model from directory: '{directory}' ")
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    



def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

    
model.config.use_cache = False
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj","up_proj", "down_proj","lm_head"],
    task_type="CAUSAL_LM"
)

base_model = get_peft_model(model, lora_config)
print_trainable_parameters(model)



def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Context:{example['context'][i][:MAX_CONTEXT]}\n\n ### Question: {example['instruction'][i]}\n ### Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts



def map_at_3(predictions, labels):
    map_sum = 0
    pred = np.argsort(-1*np.array(predictions),axis=1)[:,:3]
    for x,y in zip(pred,labels):
        z = [1/i if y==j else 0 for i,j in zip([1,2,3],x)]
        map_sum += np.sum(z)
    return map_sum / len(predictions)

def compute_metrics(p):
    predictions = p.predictions.tolist()
    labels = p.label_ids.tolist()
    return {"map@3": map_at_3(predictions, labels)}




## LOAD DATA
# df = pd.read_csv('input_data/train.csv')
file_name = 'train_folds_article_context'
print(f"TRAINING on:{file_name} dataset")
df = pd.read_csv(f'input_data/{file_name}.csv')
df['instruction'] = df['prompt'] + ' A: ' + df['A'] + ' B: ' + df['B'] + ' C: ' + df['C'] + ' D: ' + df['D'] + ' E: ' + df['E']


train_set, valid_set = df.loc[:45000], df.loc[45001:]


data_loaded = Dataset.from_pandas(train_set)
data_valid = Dataset.from_pandas(valid_set)

data_loaded = Dataset.from_pandas(df)


# training_args = TrainingArguments(
#     output_dir=f"output/{model_name}", 
#     per_device_train_batch_size=2,
#     per_device_eval_batch_size=8,
#     gradient_accumulation_steps=8,
#     num_train_epochs=2,
#     learning_rate=2e-4,
#     logging_steps=20,
#     logging_strategy="steps",
#     max_steps=400,
#     optim="paged_adamw_32bit",
#     fp16=False,
#     evaluation_strategy='steps',
#     eval_steps=25,
#     save_strategy="steps",
#     save_steps=25,
#     # load_best_model_at_end=False,
#     metric_for_best_model='map@3',
#     run_name="baseline-mistral-f32_c"
# )

training_args = TrainingArguments(
    output_dir=f"output/{model_name}", 
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=20,
    evaluation_strategy='steps',
    eval_steps=200,
    save_strategy="steps",
    save_steps=25,
    logging_strategy="steps",
    max_steps=600,
    optim="paged_adamw_32bit",
    fp16=False,
    run_name="baseline-mistral-f32_c_all45k"
)

# supervised_finetuning_trainer = SFTTrainer(
#     base_model,
#     train_dataset=data_loaded,
#     args=training_args,
#     tokenizer=tokenizer,
#     formatting_func=formatting_prompts_func,
#     peft_config=lora_config,
#     # dataset_text_field="text",
#     max_seq_length=4096,
#     eval_dataset=data_valid,
#     compute_metrics = compute_metrics,
#     data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer,
#                                                   response_template="Answer:")
# )

supervised_finetuning_trainer = SFTTrainer(
    base_model,
    train_dataset=data_loaded,
    args=training_args,
    tokenizer=tokenizer,
    formatting_func=formatting_prompts_func,
    peft_config=lora_config,
    # dataset_text_field="text",
    max_seq_length=4096,
    eval_dataset=data_valid,
    data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer,
                                                  response_template="Answer:")
)


supervised_finetuning_trainer.train()

supervised_finetuning_trainer.save_model('output/baseline_mistral_c_all45k/')