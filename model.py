import datasets
from datasets import load_dataset, Dataset, DatasetDict
import nltk
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.checkpoint
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig, DataCollatorForSeq2Seq
from transformers.modeling_outputs import Seq2SeqLMOutput
import sys
sys.path.append('/home/ubuntu/Review-Summerization/helper_function.py')
from helper_function import set_seeds, print_gpu_utilization
import os

os.environ['TORCH_USE_CUDA_DSA'] = '1'
PATH = 'data-csv/clean_data.csv'
SEED = 666



# ============== seed ============== #

set_seeds(SEED)
# ==================================== check gpu memory
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cpu'
torch.cuda.empty_cache()

print_gpu_utilization()

# ==================================== load data
# N = 5000  # None if no subset
N = None
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
decoder_max_length = 64
BATCH_SIZE = 4

tokenizer = AutoTokenizer.from_pretrained(ckpt, model_max_length=512)

if ckpt in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "summarize: "
else:
    prefix = ""


def tokenize_function(entry):
    inputs = [prefix + doc for doc in entry['text']]
    model_inputs = tokenizer(inputs,
                             padding='max_length',
                             truncation=True,
                             max_length=encoder_max_length,
                             return_tensors='pt'
                             )
    # print(model_inputs)
    # print(inputs)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(entry["summary"],
                           padding='max_length',
                           truncation=True,
                           max_length=decoder_max_length,
                           return_tensors='pt'
                           )
        # print(labels)

    token = labels['input_ids']
    token[token == tokenizer.pad_token_id] = -100
    model_inputs['labels'] = token
    # global GLOBAL_TEST
    # GLOBAL_TEST = token
    model_inputs["decoder_input_ids"] = token.clone()
    model_inputs['decoder_attention_mask'] = labels['attention_mask']
    return model_inputs


data_tokenized = data_dict.map(
    tokenize_function,
    batched=True,
    remove_columns=["text", "summary"]
)

data_tokenized.set_format(type='torch', columns=["input_ids", "attention_mask", "labels", "decoder_input_ids",
                                                 "decoder_attention_mask"])
data_collator = DataCollatorForSeq2Seq(tokenizer)

# ============================================================================================================
class MLPLayer(nn.Module):
    def __init__(self, ckpt, summary_length):
        super(MLPLayer, self).__init__()

        # T5 base
        self.model = model = AutoModelForSeq2SeqLM.from_pretrained(ckpt,
                                                                   config=AutoConfig.from_pretrained(
                                                                       ckpt,
                                                                       output_attention=True,
                                                                       output_hidden_states=True))


        # self.final_layer =
        self.dropout = nn.Dropout(0.1)
        # model.config.hidden_size =768
        self.decoder1 = nn.Linear(32128, 384)
        self.act = nn.GELU()
        self.decoder2 = nn.Linear(384, summary_length)
        self.summary_length = summary_length

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None):
        # outputs = self.model.generate(input_ids=input_ids,
        #                               attention_mask=attention_mask,
        #                               length_penalty=2,
        #                               do_sample=True,
        #                               temperature=0.7,
        #                               top_k=10,
        #                               # max_length=decoder_max_length
        #                               )
        global GLOBAL_TEST1
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask)
        # last_hidden_state = outputs[0]
        last_hidden_state = outputs.last_hidden_states
        print("last hidden:", last_hidden_state.shape)
        # GLOBAL_TEST = outputs
        sequence_output = self.dropout(last_hidden_state.to(torch.float))[:,0,:].reshape(-1, 512)
        # sequence_output = self.dropout(last_hidden_state.to(torch.float))
        # print("dropout:", sequence_output.shape)

        # GLOBAL_TEST= sequence_output
        sequence_output = self.act(self.decoder1(sequence_output))
        summary_logits = self.decoder2(sequence_output)
        # GLOBAL_TEST1 = summary_logits

        loss = None
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(summary_logits.view(-1, self.summary_length), labels.view(-1))

        return Seq2SeqLMOutput(loss=loss, logits=summary_logits, decoder_hidden_states=outputs.decoder_hidden_states,
                               decoder_attentions=outputs.decoder_attentions)


class CNNLayer(nn.Module):
    def __init__(self, ckpt, summary_length):
        super(CNNLayer, self).__init__()

        # T5 base
        self.model = model = AutoModelForSeq2SeqLM.from_pretrained(ckpt,
                                                                   config=AutoConfig.from_pretrained(
                                                                       ckpt,
                                                                       output_attention=True,
                                                                       output_hidden_states=True))

        self.conv = nn.Conv1d(in_channels=1, out_channels=256, kernel_size=7)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=3)

        self.dropout = nn.Dropout(0.1)

        self.clf1 = nn.Linear(256 * 254, 256)
        self.clf2 = nn.Linear(256, summary_length)
        self.summary_length = summary_length

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, labels=None, decoder_attention_mask=None):

        global GLOBAL_TEST1
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                             decoder_attention_mask=decoder_attention_mask)
        x = self.dropout(outputs[0])
        x = x[:, 0, :]
        # x = x.permute(0, 2, 1)
        x = x.reshape(BATCH_SIZE, 1, 768)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        # x = x.view(-1,)
        x = x.reshape(BATCH_SIZE, 256 * 254)
        x = self.clf1(x)
        x = self.relu(x)
        x = self.dropout(x)
        summary_logits = self.clf2(x)

        loss = None
        if labels is not None:
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(summary_logits.view(-1, self.summary_length), labels.view(-1))

        return Seq2SeqLMOutput(loss=loss, logits=summary_logits, decoder_hidden_states=outputs.decoder_hidden_states,
                               decoder_attentions=outputs.decoder_attentions)
# ============================================================================================================
# dataloader


train_dataloader = DataLoader(
    data_tokenized['train'], shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator
)

val_dataloader = DataLoader(
    data_tokenized['valid'], shuffle=True, collate_fn=data_collator
)

# train
from transformers import AdamW, get_scheduler

head = 'MLP'
# ckpt = "/home/ubuntu/Review-Summerization/model/T5Top2"
if head == 'MLP':
    cus_model = MLPLayer(ckpt=ckpt, summary_length=decoder_max_length).to(device)
else:
    cus_model = CNNLayer(ckpt=ckpt, summary_length=decoder_max_length).to(device)
# model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
# model = model.to(device)
# encoder = model.get_encoder()

# freeze self-attention layer of encoder
# freeze_layers = [cus_model.model.encoder.block[i].layer[0] for i in range(len(cus_model.model.encoder.block) - 1)]
# freeze self-attention layer of decoder
# freeze_layers.extend([cus_model.model.decoder.block[i].layer[0] for i in range(len(cus_model.model.decoder.block) - 1)])
# freeze cross-attention layer
# freeze_layers.extend([cus_model.model.decoder.block[i].layer[1] for i in range(len(cus_model.model.decoder.block) - 1)])

# for module in freeze_layers:
#     for param in module.parameters():
#         param.requires_grad = False
# for param in cus_model.model.parameters():
#         param.requires_grad = False

# Enable gradient checkpointing for each layer in the T5Stack
# model.model.encoder.block = torch.nn.ModuleList([torch.utils.checkpoint.checkpoint(layer) for layer in model.model.encoder.block])


optimizer = AdamW(cus_model.parameters(), lr=2e-5)
num_epoch = 3
num_training_steps = num_epoch * len(train_dataloader)
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps=0,  # baseline
    num_training_steps=num_training_steps,
)

# evaluation
metric = datasets.load_metric("rouge")


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels = np.where(labels_ids != -100, labels_ids, tokenizer.pad_token_id)
    label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Rouge expects a newline after each sentence
    pred_str = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in pred_str]
    label_str = ["\n".join(nltk.sent_tokenize(label.strip())) for label in label_str]

    result = {}
    result['rouge1'] = \
    metric.compute(predictions=pred_str, references=label_str, use_stemmer=True, rouge_types=["rouge1"])["rouge1"].mid
    result['rouge2'] = \
    metric.compute(predictions=pred_str, references=label_str, use_stemmer=True, rouge_types=["rouge2"])["rouge2"].mid
    result['rougeL'] = \
    metric.compute(predictions=pred_str, references=label_str, use_stemmer=True, rouge_types=["rougeL"])["rougeL"].mid

    return {k: round(v, 4) * 100 for k, v in result.items()}


# train
print_gpu_utilization()
from tqdm.auto import tqdm

progress_bar_train = tqdm(range(num_training_steps))
progress_bar_eval = tqdm(range(num_epoch))

gradient_accumulation_steps = 16


print("start training:")

for epoch in range(num_epoch):
    cus_model.train()
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        # print(batch)
        # input_ids = batch['input_ids']
        # print("input_ids shape: ", input_ids.shape)
        # attention_mask = batch['attention_mask']
        # print("attention_mask shape: ", input_ids.shape)
        # labels = batch['labels']
        # print("labels shape: ", labels.shape)
        # inputs = (input_ids, attention_mask)
        # Enable gradient checkpointing during the forward pass
        # with torch.backends.cudnn.flags(enabled=False):
        #     outputs = torch.utils.checkpoint.checkpoint(model.forward, *inputs)
        with torch.enable_grad():
            outputs = cus_model(**batch)
        loss = outputs.loss
        print("loss:", loss)
        # loss = outputs.loss / gradient_accumulation_steps
        loss.backward()
        # if (i + 1) % gradient_accumulation_steps == 0:
        #     optimizer.step()
        #     optimizer.zero_grad()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar_train.update(1)

    # eval
    cus_model.eval()
    for batch in val_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = cus_model(**batch)

        logit = outputs.logits
        predictions = torch.argmax(logit, dim=-1)

        metric.add_batch(predictions=predictions, references=batch['labels'])
        progress_bar_eval.update(1)

    print(metric.compute())

# post training evaluation
cus_model.eval()

test_dataloader = DataLoader(
    data_tokenized['test'], batch_size=BATCH_SIZE, collate_fn=data_collator
)

print('test:')
for batch in test_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = cus_model(**batch)

    logit = outputs.logits
    predictions = torch.argmax(logit, dim=-1)
    metric.add_batch(predictions=predictions, references=batch['labels'])

print(metric.compute())

cus_model.save('./model/T5Top2')