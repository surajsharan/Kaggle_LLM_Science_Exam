import os
import pathlib
import torch
import numpy as np
import random
import multiprocessing
import wandb



def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def optimal_num_of_loader_workers():
    num_cpus = multiprocessing.cpu_count()
    num_gpus = torch.cuda.device_count()
    optimal_value = min(num_cpus, num_gpus*4) if num_gpus else num_cpus - 1
    return optimal_value


def wand_config(config=None):
    w_config={}
    for name ,values in vars(config).items():
        if name.isupper():
            w_config[name] = values  
    return w_config


try:
    from apex import amp
    APEX_INSTALLED = True
except ImportError:
    APEX_INSTALLED = False


# Tracking
# EXPERIMENT_NAME = "longformer"
EXPERIMENT_NAME = "deberta_all100k1_"
CHANGE ="w_context_debertaformat"
RUN_PARALLEL = True




# PATH = str(pathlib.Path().absolute())+''.join('output/'+EXPERIMENT_NAME)
PATH = ''.join('output/'+EXPERIMENT_NAME+CHANGE)
pathlib.Path(PATH).mkdir(parents=True, exist_ok=True)
SEED = fix_all_seeds(1993)
# os.environ["CUDA_VISIBLE_DEVICES"]= "2,4,8,11,12,14"
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DEVICE = torch.device('cuda')

NUM_WORKERS = torch.cuda.device_count() #optimal_num_of_loader_workers()
NJOBS = 4
FP16 = False #True if APEX_INSTALLED else False
FP16_OPT_LEVEL = "O1"


## Path 


# # Model
MODEL_TYPE = "deberta-v3-large"
# MODEL_PATH = '../models/potsawee/longformer-large-4096-answering-race/'
# CONFIG_NAME = '../models/potsawee/longformer-large-4096-answering-race/'


MODEL_PATH = '../models/microsoft/deberta-v3-large/'
CONFIG_NAME = '../models/microsoft/deberta-v3-large/'


INFERENCE_PATH = "output/deberta_all100k_w_context_debertaformat/checkpoint-fold-0/" 
INFERENCE_MODEL_PATH = "output/deberta_all100k_w_context_debertaformat/checkpoint-fold-0/pytorch_model.bin"


MODEL_OUT_NAME = "model.bin"

# Tokeinzer
# TOKENIZER = '../models/potsawee/longformer-large-4096-answering-race/'
TOKENIZER =  '../models/microsoft/deberta-v3-large/'

N_LAST_HIDDEN = 12
MAX_INPUT = 768 #2500


GRADIENT_ACC_STEPS = 1
EARLY_STOPPING = 3

# Training
# TRAINING_FILE = "../input_data/train_folds.csv"
# TRAINING_FILE = "../input_data/train_folds_article_context.csv" 
TRAINING_FILE = "../input_data/train_folds_article_context_70+40.csv" 

# TESTING_FILE = "../input_data/validation_data/master_validation.csv"
TESTING_FILE = "../input_data/validation_data/master_validation_data_article_context_corrected.csv"
EPOCHS = 15

USE_AWP = False
AWP_START_EPOCH = 1
TRAIN_BATCH_SIZE = 12
VALID_BATCH_SIZE = 2
LOGGING_STEPS = 4


# Optimizer
OPTIMIZER = 'AdamW'
LEARNING_RATE = 1e-5 #3e-5 # 1.5e-5
WEIGHT_DECAY = 1e-2
EPSILON = 1e-8
MAX_GRAD_NORM = 1.0

# Scheduler
DECAY_NAME =  'cosine-warmup'
WARMUP_RATIO = 0.1
