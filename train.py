import config
import engine
import dataset
from model import DebertaModel, LongformerModel
from engine import AWP
import preprocessing
import pandas as pd
import numpy as np
import os
import math
import gc
from transformers import AutoTokenizer, AutoConfig
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
from apex import amp
import wandb
import warnings
import itertools
warnings.filterwarnings("ignore", category=UserWarning)
import logging

logging.getLogger("transformers").setLevel(logging.ERROR)


from collections import OrderedDict
# torch.autograd.set_detect_anomaly(True)




from datasets import Dataset
from dataclasses import dataclass
from transformers import LongformerTokenizer, LongformerForMultipleChoice
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoModel
from transformers import get_cosine_schedule_with_warmup
from transformers import EarlyStoppingCallback




def run(fold):
    
    
    # 1. Start a new run
    wandb.init(project="kllm", entity='surajsharan',config=config.wand_config(config))


    train_folds = pd.read_csv(config.TRAINING_FILE)

    # tokenizer = LongformerTokenizer.from_pretrained(config.TOKENIZER)
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER)
    

    # test- training
    # train_set, valid_set = train_folds[train_folds['kfold']
    #                                    != fold], train_folds[train_folds['kfold'] == fold]
    
    train_set, valid_set = train_folds.loc[:100000], train_folds.loc[100001:]
    
    train_set.reset_index(drop=True,inplace=True)
    valid_set.reset_index(drop=True,inplace=True)
    
    print(
        f'Training data : {train_set.shape} , Validation data :{valid_set.shape}')
    
    train_ds = dataset.LlmseDataset(train_set, tokenizer, is_train=True, aug_prob=1.0)
    val_ds = dataset.LlmseDataset(valid_set, tokenizer, is_train=False)

    data_collator = dataset.DataCollatorForMultipleChoice(tokenizer=tokenizer)
    train_dataloader = DataLoader(
        train_ds, 
        batch_size=config.TRAIN_BATCH_SIZE, 
        shuffle=True, 
        collate_fn=data_collator,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    valid_dataloader = DataLoader(
        val_ds, 
        batch_size=config.VALID_BATCH_SIZE, 
        shuffle=False, 
        collate_fn=data_collator,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=False
    )

    
    
    gc.collect()
    
    # Intialize Model
    device = config.DEVICE
    model_config = AutoConfig.from_pretrained(config.CONFIG_NAME, output_hidden_states=True)
    # model = LongformerModel(config.MODEL_PATH, config=model_config)
    model = DebertaModel(config.MODEL_PATH, config=model_config)
    
    

    # optimizer
    optimizer = engine.get_optimizer(model, type="i")
    
    
    # mixed precision training with NVIDIA Apex
    if config.FP16:
        model = model.to(device)
        model, optimizer = amp.initialize(model, optimizer, opt_level=config.FP16_OPT_LEVEL)
    
    
    if config.RUN_PARALLEL :
        print('Model pushed to {} GPU(s), type {}.'.format(
            torch.cuda.device_count(), 
            torch.cuda.get_device_name(0))
        )
        model = nn.DataParallel(model)
#         model = model.cuda() 
        
    model.to(device)
    
    # scheduler
    
    num_training_steps = math.ceil(
        len(train_dataloader) // config.GRADIENT_ACC_STEPS) * config.EPOCHS
    if config.WARMUP_RATIO > 0:
        num_warmup_steps = int(config.WARMUP_RATIO * num_training_steps)
    else:
        num_warmup_steps = 0
    print(
        f"Total Training Steps: {num_training_steps}, Total Warmup Steps: {num_warmup_steps}")

    scheduler = engine.get_scheduler(
        optimizer, num_warmup_steps, num_training_steps)

    
    # Early stopping
    output_dir = os.path.join(config.PATH, f"checkpoint-fold-{fold}")

    early_stopping = engine.EarlyStopping(
        patience=config.EARLY_STOPPING, verbose=True, output_path=output_dir, tokenizer=tokenizer, model_config=model_config,metric="loss")
    
    awp = AWP(model, optimizer, adv_lr=0.001, adv_eps=0.001)
    loss_fct = nn.CrossEntropyLoss(label_smoothing=0.01)
    # loss_fct = nn.MarginRankingLoss(margin=0.5)
    
    
    wandb.watch(model)
    for epoch in range(config.EPOCHS):
        train_loss = engine.train_fn(
            train_dataloader, model, optimizer, device, scheduler, awp, epoch, loss_fct)
        
        
        test_loss, predictions, true_vals = engine.evaluate(
            valid_dataloader, model, device, loss_fct)
        
        
        predictions = torch.cat(predictions).numpy()
        true_vals = torch.cat(true_vals).numpy()
        predictions = np.argsort(-predictions, 1)
        
        # predictions_as_answer_letters = np.array(list('ABCDE'))[predictions]
        # predictions_as_string = [' '.join(row) for row in predictions_as_answer_letters[:, :3]]
        map3 = preprocessing.mapk(true_vals.reshape(-1, 1), predictions.reshape(-1, 5), k=3)
            
                
        print(f"EPOCH : {epoch + 1}/{config.EPOCHS}")
        print(f"| Train Loss = {train_loss} | Valid Loss = {test_loss}| Val_map3:{map3:.4f}")
        wandb.log({'EPOCH':epoch,'Train Loss': train_loss,'Valid Loss':test_loss}, commit=False)

        
        early_stopping(val_loss= test_loss, monitor_metric_val= map3,model=model)        
        if early_stopping.early_stop:
            print("Early stopping")
            break
  
    
    del model, tokenizer, model_config , train_dataloader,valid_dataloader 
    gc.collect()
    


if __name__ == "__main__":
    
    for fold in range(0,1):  # (-1,5)
        print('-'*50)
        print(f'FOLD: {fold}')
        print('-'*50)
        run(fold)
        
        gc.collect()
    