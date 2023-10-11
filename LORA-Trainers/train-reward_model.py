
import os 
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.nn.functional as F
import bitsandbytes as bnb


from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel, PeftConfig, TaskType, PeftModelForSequenceClassification

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig,AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, RewardTrainer
from langchain.prompts import PromptTemplate
from transformers import TrainingArguments
from datasets import load_dataset
from datasets import Dataset



warnings.filterwarnings("ignore", category=UserWarning)



model_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
causal = False

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts

def _get_model_tokeinzer(path, causal=True):
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
    if causal:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            # low_cpu_mem_usage=True,
            device_map="auto"
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(path,num_labels=1)
    
    return tokenizer, model
    


directory = f"models/{model_name}"

if os.path.exists(directory):
    print(f"Loading Model from directory: '{directory}' ")
    tokenizer, model = _get_model_tokeinzer(directory,causal)
else:
    print(f"Downloading Model from huggingface: '{model_name}' ")
    tokenizer, model = _get_model_tokeinzer(model_name,causal)
    # Save the model weights to disk
    os.makedirs(directory, exist_ok=True)
    print(f"Saving the Model from directory: '{directory}' ")
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    
    
model.config.use_cache = False
tokenizer.pad_token = tokenizer.eos_token

def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_j = tokenizer(chosen, truncation=True)
        tokenized_k = tokenizer(rejected, truncation=True)

        new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_k["attention_mask"])

    return new_examples

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none",
    layers_to_transform=[18,19,20,21,22,23],
    target_modules=["query_proj", "key_proj" ,"value_proj"],
    task_type=TaskType.SEQ_CLS
)

base_model = get_peft_model(model, lora_config)
print(base_model.print_trainable_parameters())


## LOAD DATA
# df = pd.read_csv('input_data/train.csv')
file_name = 'reward_model'
print(f"TRAINING on:{file_name} dataset")
df = pd.read_csv(f'input_data/{file_name}.csv')

df = df.sample(frac=1.0, random_state=1993)
train_df, test_df =  train_test_split(df,test_size=0.33, random_state=42)


train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
)

test_dataset = test_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
)



training_args = TrainingArguments(
    output_dir=f"output/{model_name}",
    overwrite_output_dir = True,
    warmup_ratio=0.1,
    lr_scheduler_type='cosine',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    remove_unused_columns=False,
    optim="adafactor",
    logging_steps=250,
    eval_steps=250,
    evaluation_strategy='steps',
    load_best_model_at_end=True,
    save_total_limit = 2,
    fp16=True,
    bf16=False,
    weight_decay=0.01,
    report_to="none",
)


trainer = RewardTrainer(
    model=base_model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    peft_config=lora_config,
    max_length=2500,
)


trainer.train()

trainer.save_model('output/masterdata-reward-model/')