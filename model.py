
from transformers import AutoModel, AutoModelForMultipleChoice
from transformers import LongformerTokenizer, LongformerForMultipleChoice

import torch
import torch.nn as nn
import config as configuration
import numpy as np
from sklearn import metrics


class LongformerModel(nn.Module):
    def __init__(self, modelname_or_path, config, dropout=0.2, pretrained=True):
        super().__init__()

        # Transformer
        self.config = config
        if pretrained:
            self.transformer = LongformerForMultipleChoice.from_pretrained(modelname_or_path, config=config)
        else:
            self.transformer = LongformerForMultipleChoice.from_config(config)
        


    def _init_weights(self, module, config):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, labels=None):
        out = self.transformer(input_ids, attention_mask=attention_mask)
        logits = out['logits']  # batch_size x max_length (512) x 768
        
        return logits






class DebertaModel(nn.Module):
    def __init__(self, modelname_or_path, config, dropout=0.2, pretrained=True):
        super().__init__()

        # Transformer
        self.config = config
        if pretrained:
            self.transformer = AutoModelForMultipleChoice.from_pretrained(modelname_or_path, config=config)
        else:
            self.transformer = AutoModelForMultipleChoice.from_config(config)
        
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.output = nn.Linear(self.config.hidden_size, 1)
        #self.fc_dropout = nn.Dropout(dropout)
        #self.fc = nn.Linear(config.hidden_size, 1)
        #self._init_weights(self.fc, self.config)

    def _init_weights(self, module, config):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, labels=None,token_type_ids=None):
        # out = self.transformer(input_ids, attention_mask, token_type_ids=token_type_ids)
        out = self.transformer(input_ids, attention_mask =attention_mask, token_type_ids=token_type_ids)
        
        logits = out['logits']  # batch_size x max_length (512) x 768
#         sequence_output = out.hidden_states[-1]
#         sequence_output = self.dropout(sequence_output)

#         logits = self.output(sequence_output)

#         # Pooling across the sequence to get one scalar per choice
#         logits = torch.mean(logits, dim=1)  # [batch*num_choices, 1]

#         # Reshape to [batch, num_choices]
#         logits = logits.view(-1, 5)  # [batch, 5]
        
        
        return logits


class DebertaModel1(nn.Module):
    def __init__(self, modelname_or_path, config):
        super(LongformerModel, self).__init__()
        self.config = config
        self.model = AutoModel.from_pretrained(
            modelname_or_path, config=config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        self.dropout4 = nn.Dropout(0.4)
        self.dropout5 = nn.Dropout(0.5)
        
        self.output = nn.Linear(config.hidden_size, configuration.NUM_LABELS)
        

#         self._init_weights(self.output)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    
    
    def forward(self, ids, mask, word_ids = None,token_type_ids=None, targets=None):

        if token_type_ids:
            transformer_out = self.longformer(ids, mask, token_type_ids)
        else:
            transformer_out = self.longformer(ids, mask)
            
        sequence_output = transformer_out.last_hidden_state
        sequence_output = self.dropout(sequence_output)

        logits1 = self.output(self.dropout1(sequence_output))
        logits2 = self.output(self.dropout2(sequence_output))
        logits3 = self.output(self.dropout3(sequence_output))
        logits4 = self.output(self.dropout4(sequence_output))
        logits5 = self.output(self.dropout5(sequence_output))

        logits = (logits1 + logits2 + logits3 + logits4 + logits5) / 5
#         logits = torch.softmax(logits, dim=-1)
        loss = 0

        if targets is not None:
            loss1 = loss_func(logits1, targets, attention_mask=mask)
            loss2 = loss_func(logits2, targets, attention_mask=mask)
            loss3 = loss_func(logits3, targets, attention_mask=mask)
            loss4 = loss_func(logits4, targets, attention_mask=mask)
            loss5 = loss_func(logits5, targets, attention_mask=mask)
            loss = (loss1 + loss2 + loss3 + loss4 + loss5) / 5
            
            return logits, loss

        return logits, loss


    
## ADAPTED

class DebertaModel_stack(nn.Module):
    def __init__(self, modelname_or_path, config, dropout=0.2, pretrained=True):
        super().__init__()

        # Transformer
        self.config = config
        if pretrained:
            self.transformer = AutoModelForMultipleChoice.from_pretrained(modelname_or_path, config=config)
        else:
            self.transformer = AutoModelForMultipleChoice.from_config(config)

        self.high_dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(self.config.hidden_size * 2 , 1)
        #self.fc_dropout = nn.Dropout(dropout)
        #self.fc = nn.Linear(config.hidden_size, 1)
        #self._init_weights(self.fc, self.config)

    def _init_weights(self, module, config):
        module.weight.data.normal_(mean=0.0, std=config.initializer_range)
        if module.bias is not None:
            module.bias.data.zero_()

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.transformer(input_ids, attention_mask, token_type_ids=token_type_ids)
        # logits = out['logits']  # batch_size x max_length (512) x 768
        
        LAST_HIDDEN_LAYERS = 12

        out = out.hidden_states 
        out = torch.stack(tuple(out[-i - 1] for i in range(LAST_HIDDEN_LAYERS)), dim=0)
       
        out_mean = torch.mean(out, dim=0) 
        out_max, _ = torch.max(out, dim=0)
        out = torch.cat((out_mean, out_max), dim=-1)
       

        # Multisample Dropout: https://arxiv.org/abs/1905.09788
        logits = torch.mean(torch.stack([self.qa_outputs(self.high_dropout(out))for _ in range(5)], dim=0), dim=0)
        
        logits = logits.view(input_ids.shape[0], input_ids.shape[1], -1, 1)
        
        # Pooling across sequence to get one scalar per batch item
        logits_mean = torch.mean(logits, dim=-2)  # Shape: [batch_size, 1]

        # Remove the last singleton dimension
        logits_mean = logits_mean.squeeze(-1)  # Shape: [batch_size]
        #x = self.fc_dropout(x)
        #x = self.fc(x)

        return logits_mean
    


def loss_func(outputs, targets, attention_mask):
    loss_fct = nn.CrossEntropyLoss()

    active_loss = attention_mask.view(-1) == 1
    active_logits = outputs.view(-1, configuration.NUM_LABELS)
    true_labels = targets.view(-1)
    outputs = active_logits.argmax(dim=-1)
    idxs = np.where(active_loss.cpu().numpy() == 1)[0]
    active_logits = active_logits[idxs]
    true_labels = true_labels[idxs].to(torch.long)

    loss = loss_fct(active_logits, true_labels)
    return loss   

    


