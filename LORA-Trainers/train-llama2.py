
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



model_name = 'meta-llama/Llama-2-7b-hf'


def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = f"### Question: {example['instruction'][i]}\n ### Answer: {example['answer'][i]}"
        output_texts.append(text)
    return output_texts

def _get_model_tokeinzer(path):
    tokenizer = AutoTokenizer.from_pretrained(path)

    model = AutoModelForCausalLM.from_pretrained(
        path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
        # low_cpu_mem_usage=True,
        device_map="auto"
    )
    
    return tokenizer, model


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


directory = f"models/{model_name}"

if os.path.exists(directory):
    print(f"Loading Model from directory: '{directory}' ")
    tokenizer, model = _get_model_tokeinzer(directory)
else:
    print(f"Downloading Model from huggingface: '{model_name}' ")
    tokenizer, model = _get_model_tokeinzer(model_name)
    # Save the model weights to disk
    os.makedirs(directory, exist_ok=True)
    print(f"Saving the Model from directory: '{directory}' ")
    model.save_pretrained(directory)
    tokenizer.save_pretrained(directory)
    
    
model.config.use_cache = False
tokenizer.pad_token = tokenizer.eos_token

lora_config = LoraConfig(
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj","gate_proj","up_proj", "down_proj" ],
    task_type="CAUSAL_LM"
)

base_model = get_peft_model(model, lora_config)
print_trainable_parameters(model)



## LOAD DATA
# df = pd.read_csv('input_data/train.csv')
file_name = 'train_folds'
print(f"TRAINING on:{file_name} dataset")
df = pd.read_csv(f'input_data/{file_name}.csv')

df['instruction'] = df['prompt'] + ' A: ' + df['A'] + ' B: ' + df['B'] + ' C: ' + df['C'] + ' D: ' + df['D'] + ' E: ' + df['E']

data_loaded = Dataset.from_pandas(df)


training_args = TrainingArguments(
    output_dir=f"output/{model_name}", 
    per_device_train_batch_size=2,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=20,
    logging_strategy="steps",
    max_steps=400,
    optim="paged_adamw_32bit",
    fp16=True,
    run_name="baseline-llama2-masterdata-sft-f32"
)


supervised_finetuning_trainer = SFTTrainer(
    base_model,
    train_dataset=data_loaded,
    args=training_args,
    tokenizer=tokenizer,
    formatting_func=formatting_prompts_func,
    peft_config=lora_config,
    # dataset_text_field="text",
    max_seq_length=4096,
    data_collator=DataCollatorForCompletionOnlyLM(tokenizer=tokenizer,
                                                  response_template="Answer:")
)

supervised_finetuning_trainer.train()

supervised_finetuning_trainer.save_model('output/baseline-masterdata-llama2-lora-f32-sft/')