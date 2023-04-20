import nltk
from pynvml import *
import numpy as np
import pandas as pd
from transformers import RobertaTokenizerFast, EncoderDecoderModel, BertTokenizerFast, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset
import datasets
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

PATH = 'clean_data/test.csv'
SEED=666
TEST_RATIO = 0.2
# BATCH_SIZE = 64
BATCH_SIZE = 64
encoder_max_length = 512
decoder_max_length = 40
ckpt = 'checkpoint-1490'
# tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
# tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("t5-small")
# model = EncoderDecoderModel.from_pretrained(f"./models")
model = AutoModelForSeq2SeqLM.from_pretrained(f'./models')
model.to("cuda")
torch.cuda.empty_cache()

def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used // 1024 ** 2} MB.")


print_gpu_utilization()
metric = datasets.load_metric("rouge")

# def compute_metrics(eval_pred):
#     predictions, labels = eval_pred
#     decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
#     # Replace -100 in the labels as we can't decode them.
#     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
#     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
#
#     # Rouge expects a newline after each sentence
#     decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
#     decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
#
#     result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
#     # Extract a few results
#     result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
#
#     # Add mean generated length
#     prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
#     result["gen_len"] = np.mean(prediction_lens)
#
#     return {k: round(v, 4) for k, v in result.items()}

def compute_metrics(pred_str, label_str):
    # labels_ids = pred.label_ids
    # pred_ids = pred.predictions

    # all unnecessary tokens are removed
    # pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    # labels = np.where(labels_ids != -100, labels_ids, tokenizer.pad_token_id)
    # label_str = tokenizer.batch_decode(labels, skip_special_tokens=True)

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

    return result


# map data correctly
def generate_summary(batch):
    if ckpt in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
        prefix = "summarize: "
    else:
        prefix = ""
    inputs = [prefix + doc for doc in batch['text']]
    # Tokenizer will automatically set [BOS] <text> [EOS]
    inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=encoder_max_length, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    with torch.no_grad():
        # top-k sample
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            length_penalty=2,
            do_sample=True,
            temperature=0.7,
            top_k=10,
            max_length=decoder_max_length)
        # print(outputs)
    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch


df = pd.read_csv(PATH, header=0, index_col=[0])
# df = pd.read_csv(PATH, header=0, index_col=[0])
# data = df.iloc[-int(len(df) * TEST_RATIO):][['text', 'summary']].dropna()
# test_data = Dataset.from_pandas(data.iloc[:100], preserve_index=False)
test_data = Dataset.from_pandas(df, preserve_index=False)

results = test_data.map(generate_summary, batched=True, batch_size=BATCH_SIZE)
pred_str = results["pred"]
label_str = results["summary"]

# rouge1 = metric.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1"])["rouge1"].mid
# rouge2 = metric.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
# rougeL = metric.compute(predictions=pred_str, references=label_str, rouge_types=["rougeL"])["rougeL"].mid
# print("ROUGE 1 SCORE: ",rouge1)
# print("ROUGE 2 SCORE: ",rouge2)
# print("ROUGE L SCORE: ",rougeL)

print("result metric:")
score = compute_metrics(pred_str, label_str)
print("ROUGE 1 SCORE: ",score['rouge1'])
print("ROUGE 2 SCORE: ",score['rouge2'])
print("ROUGE L SCORE: ",score['rougeL'])
# import pandas as pd
# pd.DataFrame({'checkpoint': ckpt, 'ROUGE1':rouge1, 'ROUGE2':rouge2, 'ROUGEL':rougeL}).to_csv('output/T5_ROUGE_score.csv')
output = pd.DataFrame({'reviewText': results['text'], 'predSumary':results['pred'], 'actualSummary':results['summary']})
# output.to_csv('output/T5_summary.csv')

# T5
# ROUGE 1 SCORE:  Score(precision=0.08713425441261022, recall=0.21237246787101643, fmeasure=0.10427800024485728)
# ROUGE 2 SCORE:  Score(precision=0.049848501773376734, recall=0.09715565594106625, fmeasure=0.05576334830472214)
# ROUGE L SCORE:  Score(precision=0.08366079422697068, recall=0.20089801137103697, fmeasure=0.09940487722195637)
