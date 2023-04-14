import random

import datasets
import nltk
import numpy as np
import pandas as pd
# from accelerate import init_empty_weights
# from transformers import AutoConfig, AutoModelForCausalLM
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer

PATH = 'clean_data/train.csv'
ckpt = 't5-small'
SEED = 666

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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(f'DEVICE: {device}')

from pynvml import *


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")

print_gpu_utilization()

train_set = pd.read_csv(PATH, header=0, index_col=[0])
train_data = datasets.Dataset.from_pandas(train_set.iloc[[i for i in range(len(train_set)) if i % 5 != 0]],
                                          preserve_index=False).select(range(10000))
val_data = datasets.Dataset.from_pandas(train_set.iloc[[i for i in range(len(train_set)) if i % 5 == 0]],
                                          preserve_index=False).select(range(3000))
tokenizer = AutoTokenizer.from_pretrained(ckpt)



encoder_max_length = 512
decoder_max_length = 40
BATCH_SIZE = 16

if ckpt in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "summarize: "
else:
    prefix = ""

# def tokenize_function(entry):
#     inputs = [prefix + doc for doc in entry['text']]
#     model_inputs = tokenizer(inputs, truncation=True, max_length=encoder_max_length)
#     # print(inputs)
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(entry["summary"], truncation=True, max_length=decoder_max_length)
#
#     model_inputs['labels'] = labels['input_ids'].copy()
#     model_inputs['labels'] = [[-100 if token == tokenizer.pad_token_id else token for token in labels]
#                           for labels in model_inputs['labels']]
#
#     return model_inputs

def tokenize_function(batch):
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

train_data = train_data.map(
    tokenize_function,
    batched=True,
    batch_size=BATCH_SIZE,
    remove_columns=["text", "summary"]
)

val_data = val_data.map(
    tokenize_function,
    batched=True,
    batch_size=BATCH_SIZE,
    remove_columns=["text", "summary"]
)


model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
print_gpu_utilization()

# freeze self-attention layer of encoder
freeze_layers = [model.encoder.block[i].layer[0] for i in range(len(model.encoder.block)-1)]
# freeze self-attention layer of decoder
freeze_layers.extend([model.decoder.block[i].layer[0] for i in range(len(model.decoder.block)-1)])
# freeze cross-attention layer
freeze_layers.extend([model.decoder.block[i].layer[1] for i in range(len(model.decoder.block)-1)])

for module in freeze_layers:
    for param in module.parameters():
        param.requires_grad = False



training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    fp16=True,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    output_dir="./checkpoint/bert_pre",
    # logging_steps=2,
    # logging_steps=10,
    # do_train=True,
    # do_eval=True,
    # save_steps=10,
    # eval_steps=500,
    # logging_steps=1000,
    # save_steps=500,
    # warmup_steps=500,
    save_total_limit=3,
    overwrite_output_dir=True,
)


seq2seq_collator = DataCollatorForSeq2Seq(tokenizer, model=model, return_tensors='tf')

metric = datasets.load_metric("rouge")


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    data_collator=seq2seq_collator,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)
trainer.train()