import config
import engine
import dataset
from model import DebertaModel
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
from torch.nn.functional import softmax
from torch.utils.data import DataLoader
import copy
# from apex import amp
import wandb
import warnings
import itertools
warnings.filterwarnings("ignore", category=UserWarning)


from collections import OrderedDict


from transformers import AutoTokenizer, AutoConfig
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from transformers import AutoModelForMultipleChoice, TrainingArguments, Trainer, AutoModel




def run():
    
    valid_set = pd.read_csv(config.TESTING_FILE)
    
    # train_folds.answers = train_folds.answers.apply(eval)
    
#     train_folds['context'] = train_folds['context'].apply(lambda x : " ".join(x.split()) )
    # valid_set = valid_set.head(1000)
    tokenizer = AutoTokenizer.from_pretrained(config.INFERENCE_PATH)

    valid_set.reset_index(drop=True,inplace=True)
    
    print(
        f' Validation data :{valid_set.shape}')
    
    val_ds = dataset.LlmseDataset(valid_set, tokenizer, is_train=False)

    data_collator = dataset.DataCollatorForMultipleChoice(tokenizer=tokenizer)
    
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
    model_config = AutoConfig.from_pretrained(config.MODEL_PATH)
    model = DebertaModel(config.MODEL_PATH, config=model_config)
    model = nn.DataParallel(model)
    
    
    # mixed precision training with NVIDIA Apex
    # if config.FP16:
        # model, optimizer = amp.initialize(model, optimizer, opt_level=config.FP16_OPT_LEVEL)
    
        
    model.load_state_dict(torch.load(config.INFERENCE_MODEL_PATH ,map_location=torch.device('cuda')));
    # model = nn.DataParallel(model)
    model.to(device)
    
    loss_fct = nn.CrossEntropyLoss(label_smoothing=0.01)
    

    _, predictions, true_vals = engine.evaluate(
        valid_dataloader, model, device, loss_fct, inference=True)

    
    predictions = torch.cat(predictions,dim=0)
    true_vals = torch.cat(true_vals).numpy()
    
    map3 = preprocessing.mapk(true_vals.reshape(-1, 1), np.argsort(-predictions.numpy(), 1).reshape(-1, 5), k=3)
    print( f'Val_map3:{map3:.4f}')
    
    all_probabilities = softmax(predictions,dim=1)
    test_predictions = all_probabilities.numpy()
    
    predictions_as_ids = np.argsort(-test_predictions, 1)

    predictions_as_answer_letters = np.array(list('ABCDE'))[predictions_as_ids]

    valid_set['probabilities'] = valid_set.apply(lambda row: test_predictions[row.name].tolist(), axis=1)
    valid_set['prediction'] = [' '.join(row) for row in predictions_as_answer_letters[:, :3]]

    m = preprocessing.MAP_at_3(valid_set.prediction.values, valid_set.answer.values)
    print( 'CV MAP@3 =',m )
    
    
    valid_set.to_csv('validation_data_fold0_80k.csv',index=False)
    
    del model, tokenizer, model_config , valid_dataloader 
    gc.collect()
    


if __name__ == "__main__":
    run()
    gc.collect()
