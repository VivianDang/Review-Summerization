import random
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import datasets
import transformers
import pandas as pd
from datasets import Dataset
from fairseq import utils

from transformers import DataCollatorForSeq2Seq, EncoderDecoderModel, Seq2SeqTrainingArguments, Seq2SeqTrainer
# from seq2seq_trainer import Seq2SeqTrainer
from transformers import TrainingArguments
from dataclasses import dataclass, field
from typing import Optional

# configuration

SEED = 666
PATH = 'data-csv/clean_data.csv'
TEST_RATIO = 0.2
BATCH_SIZE = 1
# encoder_max_length = 70
# decoder_max_length = 10
encoder_max_length = 512
decoder_max_length = 64


# ============== seed ============== #
def set_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


set_seeds(SEED)
print('random seed: ', SEED)


# from pynvml import *
#
#
# def print_gpu_utilization():
#     nvmlInit()
#     handle = nvmlDeviceGetHandleByIndex(0)
#     info = nvmlDeviceGetMemoryInfo(handle)
#     print(f"GPU memory occupied: {info.used//1024**2} MB.")
#
# print_gpu_utilization()

# ============== load data ============== #
def rm_long(x, max_len=1000):
    if len(x.split()) > max_len:
        return ""
    else:
        return x


def map_to_length(x):
    x["txt_len"] = len(tokenizer(x["text"]).input_ids)
    x["txt_longer_512"] = int(x["txt_len"] > 512)
    x["summary_len"] = len(tokenizer(x["summary"]).input_ids)
    x["summary_longer_64"] = int(x["summary_len"] > 64)
    x["summary_longer_128"] = int(x["summary_len"] > 128)
    return x


# df = pd.read_csv(PATH, header=0, index_col=[0]).iloc[:200000]
# df = pd.read_csv(PATH, header=0, index_col=[0])
df = pd.read_csv(PATH, header=0, index_col=[0]).sample(20000, random_state=SEED)
data = df.iloc[:int(len(df) * (1 - TEST_RATIO))][['text', 'summary']]


# df['text'] = df['text'].astype(str).apply(rm_long, max_len=300)

# df['s_count'] = df['summary'].apply(lambda x: len(x.split()))
# for i in range(90, 100):
#     var = df['s_count'].values
#     var = np.sort(var, axis=None)
#     print(f'{i} percentile value is {var[int(len(var)*float(i)/100)]}')
# print(f'100{var[-1]}')
# ============== device ============== #

cuda_env = utils.CudaEnvironment()
utils.CudaEnvironment.pretty_print_cuda_env_list([cuda_env])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(f'DEVICE: {device}')



train_data = Dataset.from_pandas(data.iloc[[i for i in range(len(data)) if i % 10 != 0]], preserve_index=False)
val_data = Dataset.from_pandas(data.iloc[[i for i in range(len(data)) if i % 10 == 0]], preserve_index=False)
# data = df.iloc[-int(len(df) * TEST_RATIO):][['text', 'summary']]
# test_set = Dataset.from_pandas(data)
#
print(f'Size of training set: {train_data.shape}')
print(f'Size of validation set: {val_data.shape}')
# print(f'Size of test set: {test_set.shape}')

# construct dataset
# train_set = ReviewDataset(PATH, 'text', 'summary', TEST_RATIO, 'train')
# val_set = ReviewDataset(PATH, 'text', 'summary', TEST_RATIO, 'val')
# test_set = ReviewDataset(PATH, 'text', 'summary', TEST_RATIO, 'test')
# # construct dataloader
# train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
# val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
# test_loader = DataLoader(test_set, BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# print(f'Size of training set: {train_loader.shape}')
# print(f'Size of validation set: {val_loader.shape}')
# print(f'Size of test set: {test_loader.shape}')
# ============== tokenizer ============== #

from transformers import BertTokenizerFast

tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
# tokenizer.bos_token = tokenizer.cls_token
# tokenizer.eos_token = tokenizer.sep_token
# sample_size = 7000
# data_stats = train_data.select(range(sample_size)).map(map_to_length, num_proc=4)
# def compute_and_print_stats(x):
#   if len(x["txt_len"]) == sample_size:
#     print(
#         "Article Mean: {}, %-Articles > 512:{}, Summary Mean:{}, %-Summary > 64:{}, %-Summary > 128:{}".format(
#             sum(x["txt_len"]) / sample_size,
#             sum(x["txt_longer_512"]) / sample_size,
#             sum(x["summary_len"]) / sample_size,
#             sum(x["summary_longer_64"]) / sample_size,
#             sum(x["summary_longer_128"]) / sample_size,
#         )
#     )
#
# output = data_stats.map(
#   compute_and_print_stats,
#   batched=True,
#   batch_size=-1,
# )
# print(output)


# import matplotlib.pyplot as plt
# txt_len = [len(tokenizer.encode(s)) for s in train_data['text']]
# sum_len = [[len(tokenizer.encode(s)) for s in train_data['summary']]]
# plt.figure()
# plt.hist(txt_len, bins=20)
# plt.title('Review Token Length')
# plt.xticks(np.arange(0, 900, 100))
# plt.xlabel('length')
# plt.ylabel('count')
# plt.grid(axis='x')
# plt.show()
#
#
# plt.figure()
# plt.hist(sum_len, bins=20)
# plt.title('Summary Token Length')
# plt.xticks(np.arange(0, 100, 10))
# plt.xlabel('length')
# plt.ylabel('count')
# plt.grid(axis='x')
# plt.show()


# ============== embedding ============== #
def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  # print(batch)
  inputs = tokenizer(batch["text"], truncation=True, max_length=encoder_max_length)
  # print(inputs)
  outputs = tokenizer(batch["summary"], truncation=True, max_length=decoder_max_length)
  # print(outputs)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  # batch["decoder_input_ids"] = outputs.input_ids
  # batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

#   # because RoBERTa automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`.
#   # We have to make sure that the PAD token is ignored
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch
######################################################################
# train_data = train_data.select(range(1024))
# batch_size=64

train_data = train_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=BATCH_SIZE,
    remove_columns=["text", "summary"]
)
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"],
)
# val_data = val_data.select(range(64))
val_data = val_data.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=BATCH_SIZE,
    remove_columns=["text", "summary"]
)
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"],
)
######################################################################
#
# # training data
# train_data = train_set.map(
#     roberta_embed,
#     batched=True,
#     batch_size=BATCH_SIZE,
#     remove_columns=["text", "summary"]
# )
# train_data.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )
# # validation data
# val_data = val_set.map(
#     roberta_embed,
#     batched=True,
#     batch_size=BATCH_SIZE,
#     remove_columns=["text", "summary"]
# )
# val_data.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
# )
#
# # ============== encoder-decoder ============== #
# roberta_shared = EncoderDecoderModel.from_encoder_decoder_pretrained("roberta-base", "roberta-base", tie_encoder_decoder=True)
bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-cased", "bert-base-cased", tie_encoder_decoder=True)
# bert2bert.save_pretrained("bert2bert")
# bert2bert = EncoderDecoderModel.from_pretrained("./checkpoint/fullbert/checkpoint-900")


# ============== data collator ============== #
seq2seq_collator = DataCollatorForSeq2Seq(tokenizer, model=bert2bert)

#
bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
bert2bert.config.eos_token_id = tokenizer.sep_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id
bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size

#
# # fine-tune
# # sensible parameters for beam search
# # set decoding params
bert2bert.config.max_length = 142
bert2bert.config.min_length = 56
bert2bert.config.no_repeat_ngram_size = 1
bert2bert.config.early_stopping = True
# bert2bert.config.length_penalty = 2.0
bert2bert.config.length_penalty = 0.8
bert2bert.config.num_beams = 4

#
training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    fp16=True,
    gradient_accumulation_steps=16,
    output_dir="./checkpoint/bert",
    # logging_steps=2,
    logging_steps=10,
    # do_train=True,
    # do_eval=True,
    save_steps=10,
    eval_steps=500,
    # logging_steps=1000,
    # save_steps=500,
    warmup_steps=500,
    save_total_limit=3,
    overwrite_output_dir=True,
)

# # load rouge for validation
rouge = datasets.load_metric("rouge")
#
def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }


torch.cuda.empty_cache()
#
# training_args = Seq2SeqTrainingArguments(
#     output_dir="./checkpoint/seq2seq",
#     per_device_train_batch_size=BATCH_SIZE,
#     per_device_eval_batch_size=BATCH_SIZE,
#     predict_with_generate=True,
#     # evaluate_during_training=True,
#     evaluation_strategy='steps',
#     do_train=True,
#     do_eval=True,
#     logging_steps=2,
#     save_steps=10,
#     eval_steps=4,
#     # warmup_steps=500,
#     overwrite_output_dir=True,
#     # save_total_limit=1,
#     fp16=True,
# )
# instantiate trainer
trainer = Seq2SeqTrainer(
    model=bert2bert,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=seq2seq_collator,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)
trainer.train()

