import random
from pynvml import *
from datasets import load_dataset, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import nltk
import numpy as np
import pandas as pd
# from accelerate import init_empty_weights
# from transformers import AutoConfig, AutoModelForCausalLM
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, DataCollatorForSeq2Seq, \
    Seq2SeqTrainingArguments, \
    Seq2SeqTrainer

PATH = 'data-csv/clean_data.csv'
SEED = 666

# ==================================== check gpu memory
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(f'DEVICE: {device}')


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


print_gpu_utilization()

# ==================================== load data
N = 100000  # None if no subset
df = load_dataset('csv', data_files=PATH)
co_to_rm = df['train'].column_names
co_to_rm.remove('summary')
co_to_rm.remove('text')
df = df.remove_columns(co_to_rm)
df.set_format('torch')

if N is not None:
    df = df['train'].select(range(N))
else:
    df = df['train'][:]

# df_tv = pd.read_csv(PATH+'train.csv', header=0, index_col=[0])
# df_tst = pd.read_csv(PATH+'test.csv', header=0, index_col=[0])
# df_train, df_val = train_test_split(df_tv, test_size=0.3, random_state=SEED)
#
# data_dict = DatasetDict({
#     'train': df_train,
#     'valid': df_val,
#     'test': df_tst
# })

df_tt = df.train_test_split(test_size=0.25, seed=SEED)
df_tv = df_tt['train'].train_test_split(test_size=0.3, seed=SEED)

data_dict = DatasetDict({
    'train': df_tv['train'],
    'valid': df_tv['test'],
    'test': df_tt['test']
})

del df_tt, df_tv, df, co_to_rm
torch.cuda.empty_cache()

# ==================================== tokenize
ckpt = 't5-small'
encoder_max_length = 512
decoder_max_length = 40
BATCH_SIZE = 16

tokenizer = AutoTokenizer.from_pretrained(ckpt)

if ckpt in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "summarize: "
else:
    prefix = ""


def tokenize_function(entry):
    inputs = [prefix + doc for doc in entry['text']]
    model_inputs = tokenizer(inputs, truncation=True, max_length=encoder_max_length)
    # print(inputs)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(entry["summary"], truncation=True, max_length=decoder_max_length)

    model_inputs['labels'] = labels['input_ids'].copy()
    model_inputs['labels'] = [[-100 if token == tokenizer.pad_token_id else token for token in labels]
                              for labels in model_inputs['labels']]

    return model_inputs


data_tokenized = data_dict.map(
    tokenize_function,
    batched=True,
    remove_columns=["text", "summary"]
)

data_tokenized.set_format('torch')
data_collator = DataCollatorForSeq2Seq(tokenizer)


class SummarizationLayer(nn.Module):
    def __init__(self, ckpt, summary_length):
        super(SummarizationLayer, self).__init__()

        # T5 base
        self.model = model = AutoModelForSeq2SeqLM.from_pretrained(ckpt,
                                                                   config=AutoConfig.from_pretrained(
                                                                       ckpt,
                                                                       output_attention=True,
                                                                       output_hidden_statr=True))

        # head
        self.dropout = nn.Dropout(0.1)
        self.final_layer = nn.Linear(model.config.hidden_size, 1)
        self.summary_length = summary_length

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs[0]

        sequence_output = self.dropout(last_hidden_state)

        final_output = self.final_layer(sequence_output)

        return final_output
