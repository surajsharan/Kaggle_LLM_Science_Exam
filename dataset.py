import torch
import config
import numpy as np
from typing import Optional, Union
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy


def prepare_answering_input(
        tokenizer, # longformer_tokenizer
        question,  # str
        options,   # List[str]
        context,   # str
        max_seq_length=4096,
    ):
    c_plus_q   = context + ' ' + tokenizer.bos_token + ' ' + question
    c_plus_q_5 = [c_plus_q] * len(options)
    tokenized_examples = tokenizer(
        c_plus_q_5, options,
        max_length=max_seq_length,
        padding="longest",
        truncation=True,
        return_tensors="pt",
    )
    

    # input_ids = tokenized_examples['input_ids'].unsqueeze(0)
    # attention_mask = tokenized_examples['attention_mask'].unsqueeze(0)
    # example_encoded = {
    #     "input_ids": input_ids,
    #     "attention_mask": attention_mask,
    # }
    return tokenized_examples



def prepare_answering_input_deberta(
        tokenizer, # longformer_tokenizer
        question,  # str
        options,   # List[str]
        context,   # str
        max_seq_length=4096,
    ):
    
    first_sentence = [ "[CLS] " + context ] * 5
    second_sentences = [" #### " + question + " [SEP] " + option + " [SEP]" for option in options]
    tokenized_examples = tokenizer(first_sentence, second_sentences, truncation='only_first', 
                                  max_length=max_seq_length, add_special_tokens=False)
      

    return tokenized_examples






class LlmseDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, is_train=False, aug_prob=0.8):
        self.df = df
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.aug_prob = aug_prob
        self.option_to_index = {option: idx for idx, option in enumerate('ABCDE')}
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        example = self.df.iloc[idx]
        tokenized_example = dict()
        
        if self.is_train and torch.rand(1)<self.aug_prob:
            prm = torch.randperm(5).numpy()
            #permed_dict_i2p = option_to_index = {i: p for i, p in enumerate(prm)}
            #permed_dict_p2i = option_to_index = {p: i for i, p in enumerate(prm)}
            #permed_dict_p2a = option_to_index = {p: a for p, a in zip(prm, 'ABCDE')}
            
            # permed_a2e = np.array(['A','B','C','D','E'])[prm]
            # permed_dict_a2p = {a: p for p, a in enumerate(permed_a2e)}
            # first_sentence = [example['prompt']] * 5
            # second_sentences = [example[option] for option in permed_a2e]
            # tokenized_example = self.tokenizer(first_sentence, second_sentences, truncation=False)
            # tokenized_example['label'] = permed_dict_a2p[example['answer']]
            
            permed_a2e = np.array(['A','B','C','D','E'])[prm]
            permed_dict_a2p = {a: p for p, a in enumerate(permed_a2e)}
            
            # options = [ example[option] + " [SEP]" for option in permed_a2e] # for longformer
            options = [ example[option] for option in permed_a2e] 
            
            # tokenized_example = prepare_answering_input(tokenizer=self.tokenizer, question=example['prompt'], options=options, context= example['context'], max_seq_length = config.MAX_INPUT)
            
            tokenized_example = prepare_answering_input_deberta(tokenizer=self.tokenizer, question=example['prompt'], options=options, context= example['context'], max_seq_length = config.MAX_INPUT)
            
            # first_sentence = [ "[CLS] " + example['context'] ] * 5
            # second_sentences = [" #### " + example['prompt'] + " [SEP] " + example[option] + " [SEP]" for option in permed_a2e]
            # tokenized_example = self.tokenizer(first_sentence, second_sentences, truncation='only_first', 
                                  # max_length=config.MAX_INPUT, add_special_tokens=False)
            tokenized_example['label'] = permed_dict_a2p[example['answer']]

            
        else:
            # first_sentence = [example['prompt']] * 5
            # second_sentences = [example[option] for option in 'ABCDE']
            # tokenized_example = self.tokenizer(first_sentence, second_sentences, truncation=False)
            # tokenized_example['label'] = self.option_to_index[example['answer']]
            
#             first_sentence = [ "[CLS] " + example['context'] ] * 5
#             second_sentences = [" #### " + example['prompt'] + " [SEP] " + example[option] + " [SEP]" for option in 'ABCDE']
#             tokenized_example = self.tokenizer(first_sentence, second_sentences, truncation='only_first', 
#                                           max_length=config.MAX_INPUT, add_special_tokens=False)
            # options = [ example[option] + " [SEP]" for option in 'ABCDE'] # longformer
            options = [ example[option] for option in 'ABCDE']
            # tokenized_example = prepare_answering_input(tokenizer=self.tokenizer, question=example['prompt'], options=options, context= example['context'], max_seq_length = config.MAX_INPUT )
        
            tokenized_example = prepare_answering_input_deberta(tokenizer=self.tokenizer, question=example['prompt'], options=options, context= example['context'], max_seq_length = config.MAX_INPUT )
            
            tokenized_example['label'] = self.option_to_index[example['answer']]

        return tokenized_example
            


@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features):
        label_name = 'label' if 'label' in features[0].keys() else 'labels'
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]['input_ids'])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])
        
        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors='pt',
        )
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        batch['labels'] = torch.tensor(labels, dtype=torch.int64)
        return batch
        
        
        
