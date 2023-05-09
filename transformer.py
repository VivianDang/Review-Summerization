import datasets
import nltk
import numpy as np
import pandas as pd
import torch
import sys
sys.path.append('/home/ubuntu/Review-Summerization/helper_function.py')
from helper_function import set_seeds, print_gpu_utilization
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, \
    Seq2SeqTrainer

PATH = 'clean_data/train.csv'
# PATH = 'clean_data/item_reviews.csv'
ckpt = 't5-small'
# ckpt = 'bert-base-cased'
SEED = 666
N = 40000

set_seeds(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()
print(f'DEVICE: {device}')
print_gpu_utilization()


train_set = pd.read_csv(PATH, header=0, index_col=[0]).sample(N, random_state=SEED)
train_data = datasets.Dataset.from_pandas(train_set.iloc[[i for i in range(len(train_set)) if i % 5 != 0]],
                                          preserve_index=False)
val_data = datasets.Dataset.from_pandas(train_set.iloc[[i for i in range(len(train_set)) if i % 5 == 0]],
                                        preserve_index=False)
tokenizer = AutoTokenizer.from_pretrained(ckpt)

encoder_max_length = 1024
decoder_max_length = 80
# encoder_max_length = 512
# decoder_max_length = 64
BATCH_SIZE = 16

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


train_data = train_data.map(
    tokenize_function,
    batched=True,
    batch_size=BATCH_SIZE,
    remove_columns=["text", "summary"]
)
# train_data.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "labels"],
# )

val_data = val_data.map(
    tokenize_function,
    batched=True,
    batch_size=BATCH_SIZE,
    remove_columns=["text", "summary"]
)
# val_data.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "labels"],
# )

model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
print_gpu_utilization()


# freeze bottom layers of encoder
freeze_layers = [model.encoder.block[i] for i in range(len(model.encoder.block) - 2)]
freeze_layers.extend([model.encoder.block[i] for i in range(len(model.decoder.block) - 2)])
# freeze_layers = [model.encoder.block[i] for i in range(len(model.encoder.block))]
# freeze_layers.extend([model.encoder.block[i] for i in range(len(model.decoder.block))])
for module in freeze_layers:
    for param in module.parameters():
        param.requires_grad = False


# check layers Print the requires_grad attribute of each parameter in the model
# for name, param in model.named_parameters():
#     print(name, param.requires_grad)
model.config.use_cache = False
# model.config.length_penalty=2
# model.config.num_beams=3
# model.config.no_repeat_ngram_size=1
# model.config.early_stopping=True
model.config.max_length=decoder_max_length

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    save_strategy='epoch',
    learning_rate=5e-5,
    lr_scheduler_type='linear',
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    weight_decay=0.01,
    fp16=True,
    gradient_accumulation_steps=16,
    gradient_checkpointing=True,
    output_dir="./checkpoint/T5_longform",
    num_train_epochs=10,
    logging_strategy='steps',
    logging_steps=100,
    logging_dir='./logs',
    do_train=True,
    do_eval=True,
    # eval_steps=500,
    # save_steps=500,
    eval_steps=100,
    save_steps=100,
    warmup_steps=100,
    # warmup_steps=500,
    save_total_limit=3,
    overwrite_output_dir=True,
)

seq2seq_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

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
    # optimizers=optimizer,
    data_collator=seq2seq_collator,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)
trainer.train()
# {'train_runtime': 4381.1889, 'train_samples_per_second': 17.529, 'train_steps_per_second': 0.068, 'train_loss': 4.020676676432291, 'epoch': 3.0}
# {'eval_loss': 3.3393425941467285, 'eval_rouge1': 12.6772, 'eval_rouge2': 6.5119, 'eval_rougeL': 12.2064, 'eval_rougeLsum': 12.227, 'eval_gen_len': 11.8267, 'eval_runtime': 244.0334, 'eval_samples_per_second': 26.226, 'eval_steps_per_second': 1.639, 'epoch': 3.0}
#
# {'train_runtime': 4474.754, 'train_samples_per_second': 21.454, 'train_steps_per_second': 0.084, 'train_loss': 3.7059837900797525, 'epoch': 3.0}
# {'eval_loss': 3.0026843547821045, 'eval_rouge1': 13.3352, 'eval_rouge2': 6.9143, 'eval_rougeL': 12.9298, 'eval_rougeLsum': 12.9477, 'eval_gen_len': 10.3405, 'eval_runtime': 266.4063, 'eval_samples_per_second': 30.029, 'eval_steps_per_second': 1.877, 'epoch': 3.0}
#
# {'train_runtime': 4509.7754, 'train_samples_per_second': 21.287, 'train_steps_per_second': 0.083, 'train_loss': 3.5159979553222658, 'epoch': 3.0}
# {'eval_loss': 2.8896069526672363, 'eval_rouge1': 13.6914, 'eval_rouge2': 6.9468, 'eval_rougeL': 13.283, 'eval_rougeLsum': 13.3142, 'eval_gen_len': 9.7924, 'eval_runtime': 261.3744, 'eval_samples_per_second': 30.607, 'eval_steps_per_second': 1.913, 'epoch': 3.0}
#
# {'train_runtime': 15629.4598, 'train_samples_per_second': 20.474, 'train_steps_per_second': 0.08, 'train_loss': 3.0075174560546873, 'epoch': 10.0}
# {'eval_loss': 2.5083444118499756, 'eval_rouge1': 30.9199, 'eval_rouge2': 22.9759, 'eval_rougeL': 30.6549, 'eval_rougeLsum': 30.6498, 'eval_gen_len': 7.4021, 'eval_runtime': 357.434, 'eval_samples_per_second': 22.382, 'eval_steps_per_second': 1.399, 'epoch': 10.0}
#
#{'train_runtime': 20681.2962, 'train_samples_per_second': 15.473, 'train_steps_per_second': 0.06, 'train_loss': 3.0075174560546873, 'epoch': 10.0}
# {'eval_loss': 2.5083444118499756, 'eval_rouge1': 30.6878, 'eval_rouge2': 22.8856, 'eval_rougeL': 30.3796, 'eval_rougeLsum': 30.4104, 'eval_gen_len': 6.8791, 'eval_runtime': 745.5379, 'eval_samples_per_second': 10.731, 'eval_steps_per_second': 0.671, 'epoch': 10.0}
trainer.save_model('./checkpoint/T5')