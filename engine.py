

import config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import transformers
from transformers import AdamW
from collections import OrderedDict

from apex import amp
import os
import collections
import wandb
import gc
import warnings
import itertools
warnings.filterwarnings("ignore", category=UserWarning)

def listnet_loss(y_pred, y_true):
    # Convert the ground truth to one-hot encoding
    y_true_onehot = F.one_hot(y_true, num_classes=y_pred.size(1)).float()

    # Ensure the predictions are in the same order as y_true.
    _, indices = torch.sort(y_true_onehot, descending=True, dim=1)
    y_pred = torch.gather(y_pred, 1, indices)

    # Compute softmax over raw scores
    y_pred_softmax = F.softmax(y_pred, dim=1)
    
    # Compute cross-entropy
    loss = -torch.mean(torch.sum(y_true_onehot * torch.log(y_pred_softmax), dim=1))
    
    return loss

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, output_path=None, tokenizer=None, model_config=None,metric="loss"):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.metric = metric
        self.output_path = output_path
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.counter = 0
        self.best_score = None
        self.monitor_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.loss_ = []
        self.monitor_ = []
        self.monitor_val_min = -np.Inf

    def __call__(self, val_loss,monitor_metric_val, model):

        
        score = -val_loss
        monitor_metric = monitor_metric_val
        self.loss_.append(score)
        self.monitor_.append(monitor_metric)

        if self.best_score is None:
            self.best_score = score
            self.monitor_score = monitor_metric
            
            self.save_checkpoint(val_loss,monitor_metric_val, model)
            
        elif score > 1.2 * np.max(self.loss_) and monitor_metric == np.max(self.monitor_):
            self.best_score = score
            self.monitor_score = monitor_metric
            self.save_checkpoint(val_loss,monitor_metric_val, model)
            
            
        elif score < self.best_score :
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
            
        else:
            self.best_score = score
            self.monitor_score = monitor_metric
            self.save_checkpoint(val_loss,monitor_metric_val, model)
            self.counter = 0

    def save_checkpoint(self, val_loss,monitor_metric_val, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            if self.metric == "loss":
                print( f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
                print( f' MAP@3 increased ({self.monitor_val_min:.6f} --> {monitor_metric_val:.6f}).  Saving model ...')
            
        os.makedirs(self.output_path, exist_ok=True)

        torch.save(model.state_dict(), f"{self.output_path}/pytorch_model.bin")
        self.model_config.save_pretrained(self.output_path)
        self.tokenizer.save_pretrained(self.output_path)
        
        self.val_loss_min = val_loss
        self.monitor_val_min = monitor_metric_val
        
        print(f"Saving model checkpoint to {self.output_path}.")
        
        
    def get_best_jaccard(self):
        return self.monitor_val_min 

# Metric Logger


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.max = 0
        self.min = 1e5

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if val > self.max:
            self.max = val
        if val < self.min:
            self.min = val


def get_optimizer(model, type="s"):
    optimizer_parameters = get_optimizer_params(model, type="s")
    if config.OPTIMIZER == "AdamW":
        optimizer = AdamW(
            optimizer_parameters,
            lr=config.LEARNING_RATE,
            eps=config.EPSILON,
            correct_bias=True
        )
        return optimizer

     

def get_optimizer_params(model, type='s'):
    '''
    differential learning rate and weight decay
       s : unified lr for the whole model   
       i : differential lr for transformer and task layer
       a : differntial lr for transformer layer and task layers
    '''

    no_decay = ['bias', 'LayerNorm.bias', "LayerNorm.weight"]
    if type == 's':
        optimizer_parameters = filter(
            lambda x: x.requires_grad, model.parameters())
    elif type == 'i':
        optimizer_parameters = [
        {'params': [p for n, p in model.transformer.named_parameters() if not any(nd in n for nd in no_decay)],
         'lr': config.ENCODER_LEARNING_RATE, 'weight_decay': weight_decay},
        {'params': [p for n, p in model.transformer.named_parameters() if any(nd in n for nd in no_decay)],
         'lr': config.ENCODER_LEARNING_RATE, 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if 'transformer' not in n],
         'lr': config.ENCODER_LEARNING_RATE, 'weight_decay': 0.0}
                                ]
    elif type == 'a':
        group1 = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.']
        group2 = ['layer.4.', 'layer.5.', 'layer.6.', 'layer.7.']
        group3 = ['layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']
        group_all = ['layer.0.', 'layer.1.', 'layer.2.', 'layer.3.', 'layer.4.', 'layer.5.', 'layer.6.', 'layer.7.',
                     'layer.8.', 'layer.9.', 'layer.10.', 'layer.11.']
        optimizer_parameters = [
            {'params': [p for n, p in model.xlm_roberta.named_parameters() if
                        not any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
             'weight_decay_rate': config.WEIGHT_DECAY},
            {'params': [p for n, p in model.xlm_roberta.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group1)],
             'weight_decay_rate': config.WEIGHT_DECAY, 'lr': config.LEARNING_RATE / 2.6},
            {'params': [p for n, p in model.xlm_roberta.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group2)],
             'weight_decay_rate': config.WEIGHT_DECAY, 'lr': config.LEARNING_RATE},
            {'params': [p for n, p in model.xlm_roberta.named_parameters() if
                        not any(nd in n for nd in no_decay) and any(nd in n for nd in group3)],
             'weight_decay_rate': config.WEIGHT_DECAY, 'lr': config.LEARNING_RATE * 2.6},
            {'params': [p for n, p in model.xlm_roberta.named_parameters() if
                        any(nd in n for nd in no_decay) and not any(nd in n for nd in group_all)],
             'weight_decay_rate': 0.0},
            {'params': [p for n, p in model.xlm_roberta.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group1)], 'weight_decay_rate': 0.0,
             'lr': config.LEARNING_RATE / 2.6},
            {'params': [p for n, p in model.xlm_roberta.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group2)], 'weight_decay_rate': 0.0,
             'lr': config.LEARNING_RATE},
            {'params': [p for n, p in model.xlm_roberta.named_parameters() if
                        any(nd in n for nd in no_decay) and any(nd in n for nd in group3)], 'weight_decay_rate': 0.0,
             'lr': config.LEARNING_RATE * 2.6},
            {'params': [p for n, p in model.named_parameters() if config.MODEL_TYPE not in n],
             'lr': config.LEARNING_RATE * 20, "momentum": 0.99, 'weight_decay_rate': 0.0},
        ]
    return optimizer_parameters


def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    if config.DECAY_NAME == "cosine-warmup":
        scheduler = transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=0.5
        )
    else:
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    return scheduler



def train_fn(dataloader, model, optimizer, device, scheduler, awp, epoch, loss_fct):
    
    count = 0
    losses = AverageMeter()
    # model.zero_grad()
    model.train()
    
    #Log gradients and model parameters
    wandb.watch(model)
    
    
    with tqdm(dataloader, leave=True) as pbar:
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(pbar):                                
            # inp_ids = batch['input_ids'].to(device)
            # att_mask = batch['attention_mask'].to(device)
            # token_type_ids = batch['token_type_ids'].to(device)
            # label = batch['labels'].to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            

            if config.USE_AWP:
                if epoch >= config.AWP_START_EPOCH:
                    awp.perturb(inp_ids, att_mask, token_type_ids, label, loss_fct)
                
            y_pred = model(**batch)
            # y_pred = model(input_ids=inp_ids, 
            #                    attention_mask=att_mask, token_type_ids=token_type_ids)
    
            
#             # Compute MarginRankingLoss
#             correct_indices = batch['labels'].view(-1)
#             positive_scores = y_pred[torch.arange(y_pred.size(0)), correct_indices]
#             # Step 1: Create a mask
#             mask = torch.ones(y_pred.shape, dtype=torch.bool)
#             mask[torch.arange(y_pred.size(0)), correct_indices] = 0

#             # Step 2: Use this mask to select the scores associated with the incorrect labels
#             negative_scores_all = y_pred[mask].view(y_pred.size(0), -1)

#             # Step 3: If you just want one negative score (maximum) for each instance
#             max_negative_scores = negative_scores_all.max(dim=1).values

            
#             # We specify a tensor of -1 because we want positive_scores - negative_scores to be > margin
#             # Compute MarginRankingLoss
#             target = torch.full_like(positive_scores, -1.0).unsqueeze(-1)
#             loss = loss_fct(positive_scores.unsqueeze(-1), max_negative_scores.unsqueeze(-1), target)
            
            loss = loss_fct(y_pred, batch['labels'].view(-1))  # nn crossentropy loss

        
            if config.GRADIENT_ACC_STEPS > 1:
                    loss = loss / config.GRADIENT_ACC_STEPS
            
            
            losses.update(loss.mean().item(), batch['input_ids'].size(0))
            
            if config.FP16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.sum().backward()
            
            pbar.set_postfix(
                OrderedDict(
                    epoch=f'{epoch+(batch_idx+1)/len(dataloader):.2f}',
                    loss=f'{loss.mean().item()*config.GRADIENT_ACC_STEPS:.4f}',
                    lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                )
            )
            
            
                
            awp.restore()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM or 1e9)
            
            if batch_idx % config.GRADIENT_ACC_STEPS == 0 or batch_idx == len(dataloader) - 1:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            if batch_idx % config.LOGGING_STEPS == 0:
                # 4. Log metrics to visualize performance
                wandb.log({"train loss": loss.mean().item()})


    return losses.avg



def evaluate(dataloader, model, device, loss_fct, inference=False):
    losses = AverageMeter()
    model.eval()
    predictions, true_vals = [], []
    with torch.no_grad():
        
        if inference:
            with tqdm(dataloader, leave=False) as pbar:
                for idx, batch in enumerate(pbar):

                    # inp_ids = batch['input_ids'].to(device)
                    # att_mask = batch['attention_mask'].to(device)
                    # token_type_ids = batch['token_type_ids'].to(device)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    # label = batch['labels'].to(device)
                    
                    y_pred = model(**batch)
                    # y_pred = model(input_ids=inp_ids, 
                    #                attention_mask=att_mask, token_type_ids=token_type_ids)
                    y_pred = y_pred.to(torch.float)
    
                    true_vals.append(batch['labels'].detach().cpu())
                    predictions.append(y_pred.detach().cpu())

        else:
            with tqdm(dataloader, leave=False) as pbar:
                for idx, batch in enumerate(pbar):

                    # inp_ids = batch['input_ids'].to(device)
                    # att_mask = batch['attention_mask'].to(device)
                    # token_type_ids = batch['token_type_ids'].to(device)
                    # label = batch['labels'].to(device)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    
                    
                    y_pred = model(**batch)
                    # y_pred = model(input_ids=inp_ids, 
                    #                attention_mask=att_mask, 
                    #                token_type_ids=token_type_ids)
                    
                    loss = loss_fct(y_pred, batch['labels'].view(-1))
                    
                    
#                     # Compute MarginRankingLoss
#                     correct_indices = batch['labels'].view(-1)
#                     positive_scores = y_pred[torch.arange(y_pred.size(0)), correct_indices]
#                     # Step 1: Create a mask
#                     mask = torch.ones(y_pred.shape, dtype=torch.bool)
#                     mask[torch.arange(y_pred.size(0)), correct_indices] = 0

#                     # Step 2: Use this mask to select the scores associated with the incorrect labels
#                     negative_scores_all = y_pred[mask].view(y_pred.size(0), -1)

#                     # Step 3: If you just want one negative score (maximum) for each instance
#                     max_negative_scores = negative_scores_all.max(dim=1).values


#                     # We specify a tensor of -1 because we want positive_scores - negative_scores to be > margin
#                     target = torch.full_like(positive_scores, -1.0).unsqueeze(-1)
#                     loss = loss_fct(positive_scores.unsqueeze(-1), max_negative_scores.unsqueeze(-1), target)
        
                    losses.update(loss.mean().item(), batch['input_ids'].size(0))
                    
                    y_pred = y_pred.to(torch.float)
    
                    true_vals.append(batch['labels'].detach().cpu())
                    predictions.append(y_pred.detach().cpu())
            

    return losses.avg, predictions, true_vals


class AWP:
    def __init__(self, model, optimizer, *, adv_param='weight',
                 adv_lr=0.001, adv_eps=0.001):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def perturb(self, input_ids, attention_mask, token_type_ids, y, criterion):
        """
        Perturb model parameters for AWP gradient
        Call before loss and loss.backward()
        """
        self._save()  # save model parameters
        self._attack_step()  # perturb weights

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                grad = self.optimizer.state[param]['exp_avg']
                norm_grad = torch.norm(grad)
                norm_data = torch.norm(param.detach())

                if norm_grad != 0 and not torch.isnan(norm_grad):
                    # Set lower and upper limit in change
                    limit_eps = self.adv_eps * param.detach().abs()
                    param_min = param.data - limit_eps
                    param_max = param.data + limit_eps

                    # Perturb along gradient
                    # w += (adv_lr * |w| / |grad|) * grad
                    param.data.add_(grad, alpha=(self.adv_lr * (norm_data + e) / (norm_grad + e)))

                    # Apply the limit to the change
                    param.data.clamp_(param_min, param_max)

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.clone().detach()
                else:
                    self.backup[name].copy_(param.data)

    def restore(self):
        """
        Restore model parameter to correct position; AWP do not perturbe weights, it perturb gradients
        Call after loss.backward(), before optimizer.step()
        """
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])