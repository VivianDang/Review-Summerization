import pandas as pd
from transformers import RobertaTokenizerFast, EncoderDecoderModel, BertTokenizerFast
from torch.utils.data import Dataset
import datasets
import torch
from datasets import Dataset

PATH = 'data-csv/clean_data.csv'
SEED=666
TEST_RATIO = 0.2
# BATCH_SIZE = 64
BATCH_SIZE = 16
encoder_max_length = 512
decoder_max_length = 64
ckpt = 'checkpoint-1490'
# tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
model = EncoderDecoderModel.from_pretrained(f"./checkpoint/bert/{ckpt}")
model.to("cuda")
torch.cuda.empty_cache()
rouge = datasets.load_metric("rouge")

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

# map data correctly
def generate_summary(batch):
    # Tokenizer will automatically set [BOS] <text> [EOS]
    inputs = tokenizer(batch["text"], padding="max_length", truncation=True, max_length=encoder_max_length, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    with torch.no_grad():
        outputs = model.generate(input_ids, attention_mask=attention_mask, length_penalty=0.8, num_beams=4, max_length=decoder_max_length)
    # all special tokens including will be removed
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred"] = output_str

    return batch

df = pd.read_csv(PATH, header=0, index_col=[0]).sample(20000, random_state=SEED)
# df = pd.read_csv(PATH, header=0, index_col=[0])
data = df.iloc[-int(len(df) * TEST_RATIO):][['text', 'summary']].dropna()
test_data = Dataset.from_pandas(data.iloc[:100], preserve_index=False)
# test_data = Dataset.from_pandas(data, preserve_index=False)

results = test_data.map(generate_summary, batched=True, batch_size=BATCH_SIZE, remove_columns=["text"])
pred_str = results["pred"]
label_str = results["summary"]

rouge1 = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge1"])["rouge1"].mid
rouge2 = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid
rougeL = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rougeL"])["rougeL"].mid
print("ROUGE 1 SCORE: ",rouge1)
print("ROUGE 2 SCORE: ",rouge2)
print("ROUGE L SCORE: ",rougeL)

# import pandas as pd
# pd.DataFrame({'checkpoint': ckpt, 'ROUGE1':rouge1, 'ROUGE2':rouge2, 'ROUGEL':rougeL}).to_csv('output/ROUGE_score.csv')
output = pd.DataFrame({'reviewText': test_data['text'], 'predSumary':results['pred'], 'actualSummary':results['summary']})
# output.to_csv('output/summary_fullbert.csv')
